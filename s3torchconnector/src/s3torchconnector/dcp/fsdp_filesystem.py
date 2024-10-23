#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import io
import logging
import os
from contextlib import contextmanager
from typing import Generator, Union, Optional

from tenacity import (
    retry,
    stop_after_attempt,
    stop_after_delay,
    wait_fixed,
    wait_random,
    retry_if_exception_type,
)
from torch.distributed.checkpoint.filesystem import (
    FileSystem,
    FileSystemReader,
    FileSystemWriter,
)

from s3torchconnector import S3Checkpoint
from s3torchconnector._s3client import S3Client
from s3torchconnector._s3dataset_common import parse_s3_uri
from s3torchconnectorclient._mountpoint_s3_client import S3Exception

logger = logging.getLogger(__name__)


class S3FileSystem(FileSystem):
    def __init__(self, region: str, s3_client: Optional[S3Client] = None) -> None:
        self.path: Union[str, os.PathLike] = ""
        self.region = region
        self.client = s3_client if s3_client is not None else S3Client(region)
        self.checkpoint = S3Checkpoint(region)

    @contextmanager
    def create_stream(
        self, path: Union[str, os.PathLike], mode: str
    ) -> Generator[io.IOBase, None, None]:
        """
        Create a stream for reading or writing to S3.

        Args:
            path (Union[str, os.PathLike]): The S3 path to read or write.
            mode (str): The mode for the stream. Supports 'rb' for read mode and 'wb' for write mode.

        Yields:
            io.BufferedIOBase: A stream for reading or writing to S3.

        Raises:
            ValueError: If the mode is not 'rb' or 'wb'.
        """
        path = _path_to_str_or_pathlike(path)

        if mode == "wb":  # write mode
            logger.debug("create_stream writable for %s", path)
            with self.checkpoint.writer(path) as stream:
                yield stream
        elif mode == "rb":  # read mode
            logger.debug("create_stream readable for %s", path)
            with self.checkpoint.reader(path) as stream:
                yield stream
        else:
            raise ValueError(
                "Invalid mode argument, create_stream only supports rb (read mode) & wb (write mode)"
            )

    def concat_path(self, path: Union[str, os.PathLike], suffix: str) -> str:
        """
        Concatenate a suffix to the given path.

        Args:
            path (Union[str, os.PathLike]): The base path.
            suffix (str): The suffix to concatenate.

        Returns:
            str: The concatenated path.
        """
        logger.debug("concat paths %s and %s", path, suffix)
        path_str = os.fspath(path)
        result = os.path.join(path_str, suffix)
        return result

    def init_path(self, path: Union[str, os.PathLike]) -> Union[str, os.PathLike]:
        """
        Initialize the path for the filesystem.

        Args:
            path (Union[str, os.PathLike]): The path to initialize.

        Returns:
            Union[str, os.PathLike]: The initialized path.
        """
        logger.debug("init_path for %s", path)
        self.path = path
        return self.path

    def rename(
        self, old_path: Union[str, os.PathLike], new_path: Union[str, os.PathLike]
    ) -> None:
        """Rename an object in S3.

        This is emulated by copying it to a new path and deleting the old path. The deletion part is retried (see also
        :func:`S3FileSystem._delete_with_retry`.

        Args:
            old_path (Union[str, os.PathLike]): The current path of the object.
            new_path (Union[str, os.PathLike]): The new path for the object.

        Raises:
            S3Exception: If there is an error with the S3 client.
        """
        logger.debug("rename %s to %s", old_path, new_path)

        old_path = _path_to_str_or_pathlike(old_path)
        new_path = _path_to_str_or_pathlike(new_path)

        bucket_name, old_key = parse_s3_uri(old_path)
        _, new_key = parse_s3_uri(new_path)

        try:
            self.client.copy_object(
                src_bucket=bucket_name,
                src_key=old_key,
                dst_bucket=bucket_name,
                dst_key=new_key,
            )
            self._delete_with_retry(bucket_name, old_key)
        except S3Exception:
            logger.exception("Error renaming object in S3")

    def mkdir(self, path: Union[str, os.PathLike]) -> None:
        """No-op method for creating directories in S3 (not needed)."""
        pass

    def exists(self, path: Union[str, os.PathLike]) -> bool:
        logger.debug("exists %s", path)

        path = _path_to_str_or_pathlike(path)
        bucket, key = parse_s3_uri(path)
        try:
            self.client.head_object(bucket, key)
        except S3Exception:
            return False
        else:
            return True

    def rm_file(self, path: Union[str, os.PathLike]) -> None:
        logger.debug("remove %s", path)

        path = _path_to_str_or_pathlike(path)
        bucket, key = parse_s3_uri(path)
        try:
            self.client.delete_object(bucket, key)
        except S3Exception:
            logger.exception("Failed to remove object from S3")

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        """
        Validate the checkpoint ID.

        Args:
            checkpoint_id (Union[str, os.PathLike]): The checkpoint ID to validate.

        Returns:
            bool: True if the checkpoint ID is valid, False otherwise.
        """
        logger.debug("validate_checkpoint_id for %s", checkpoint_id)
        return FileSystem.validate_checkpoint_id(checkpoint_id)

    @retry(
        retry=retry_if_exception_type(S3Exception),
        stop=stop_after_attempt(3),
        wait=wait_fixed(3) + wait_random(0, 2),
        reraise=True,
    )
    def _delete_with_retry(self, bucket_name: str, old_key: str):
        """Wrapper around :func:`S3Client.delete_object` to retry the deletion.

        Will retry maximum 3 times, only for `S3Exception`s, and wait between retries. It will reraise the caught
        exception too."""
        self.client.delete_object(bucket_name, old_key)


class S3StorageWriter(FileSystemWriter):
    def __init__(
        self,
        region: str,
        path: Union[str, os.PathLike],
        single_file_per_rank: bool = True,
        thread_count: int = 1,
        per_thread_copy_ahead: int = 10_000_000,
        overwrite: bool = False,
    ) -> None:
        """
        Initialize an S3 writer for distributed checkpointing.

        Args:
            region (str): The AWS region for S3.
            path (Union[str, os.PathLike]): The S3 path to write checkpoints.
            single_file_per_rank (bool, optional): Whether to write a single file per rank. Defaults to True.
            thread_count (int, optional): The number of threads to use for writing. Defaults to 1.
            per_thread_copy_ahead (int, optional): The number of bytes to copy ahead per thread. Defaults to 10_000_000.
            overwrite (bool, optional): Whether to overwrite existing checkpoints. Defaults to False.
        """
        super().__init__(
            path=path,
            single_file_per_rank=single_file_per_rank,
            sync_files=False,
            thread_count=thread_count,
            per_thread_copy_ahead=per_thread_copy_ahead,
            overwrite=overwrite,
        )
        self.fs = S3FileSystem(region)
        self.path = self.fs.init_path(path)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return S3FileSystem.validate_checkpoint_id(checkpoint_id)


class S3StorageReader(FileSystemReader):
    def __init__(self, region: str, path: Union[str, os.PathLike]) -> None:
        """
        Initialize an S3 reader for distributed checkpointing.

        Args:
            region (str): The AWS region for S3.
            path (Union[str, os.PathLike]): The S3 path to read checkpoints from.
        """
        super().__init__(path)
        self.fs = S3FileSystem(region)
        self.path = self.fs.init_path(path)
        self.sync_files = False

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return S3FileSystem.validate_checkpoint_id(checkpoint_id)


def _path_to_str_or_pathlike(path: Union[str, os.PathLike]) -> str:
    return path if isinstance(path, str) else str(path)
