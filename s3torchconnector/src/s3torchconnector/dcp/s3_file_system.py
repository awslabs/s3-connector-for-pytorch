#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import io
import logging
import os
from contextlib import contextmanager
from typing import Generator, Union, Optional

from s3torchconnectorclient._mountpoint_s3_client import S3Exception
from tenacity import (
    retry,
    stop_after_attempt,
    retry_if_exception_type,
    before_sleep_log,
    after_log,
    wait_random_exponential,
)
from torch.distributed.checkpoint.filesystem import (
    FileSystemReader,
    FileSystemWriter,
    FileSystemBase,
)

from s3torchconnector._s3client import S3Client
from s3torchconnector._s3dataset_common import parse_s3_uri

logger = logging.getLogger(__name__)


class S3FileSystem(FileSystemBase):
    def __init__(self, region: str, s3_client: Optional[S3Client] = None) -> None:
        self._path: Union[str, os.PathLike] = ""
        self._client = s3_client if s3_client is not None else S3Client(region)

    @contextmanager
    def create_stream(
        self, path: Union[str, os.PathLike], mode: str
    ) -> Generator[io.IOBase, None, None]:
        """
        Create a stream for reading or writing to S3.

        Args:
            path (Union[str, os.PathLike]): The path (S3 URI) to read from or write to.
            mode (str): The mode for the stream; supports "rb" for read mode and "wb" for write mode.

        Yields:
            io.IOBase: A stream for reading from or writing to S3.

        Raises:
            ValueError: If the mode is neither "rb" nor "wb".
        """
        bucket, key = parse_s3_uri(str(path))

        if mode == "wb":
            logger.debug("create_stream: write to %s", path)
            with self._client.put_object(bucket, key) as stream:
                yield stream
        elif mode == "rb":
            logger.debug("create_stream: read from %s", path)
            with self._client.get_object(bucket, key) as stream:
                yield stream
        else:
            raise ValueError(
                f'Invalid {mode=} argument: `create_stream` only supports "rb" (read) or "wb" (write) modes'
            )

    def concat_path(self, path: Union[str, os.PathLike], suffix: str) -> str:
        """
        Concatenate a suffix to the given path (S3 URI).

        Args:
            path (Union[str, os.PathLike]): The base path.
            suffix (str): The suffix to concatenate.

        Returns:
            str: The concatenated path (S3 URI).
        """
        logger.debug("concat_path: %s to %s", path, suffix)
        path_str = os.fspath(path)  # FIXME: handle properly S3 URIs
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
        logger.debug("init_path: %s", path)
        self._path = path
        return self._path

    def rename(
        self, path: Union[str, os.PathLike], new_path: Union[str, os.PathLike]
    ) -> None:
        """Rename an object in S3.

        This is emulated by copying it to a new path and deleting the old one. The deletion part is retried (see also
        :func:`S3FileSystem._delete_with_retry`).

        Args:
            path (Union[str, os.PathLike]): The current path (URI) of the object.
            new_path (Union[str, os.PathLike]): The new path (URI) for the object.

        Raises:
            ValueError: If the old and new paths (URIs) point to different buckets.
            S3Exception: If there is an error with the S3 client.
        """
        logger.debug("rename: %s to %s", path, new_path)

        bucket, key = parse_s3_uri(str(path))
        new_bucket, new_key = parse_s3_uri(str(new_path))

        if bucket != new_bucket:
            raise ValueError(
                f"Source and destination buckets cannot be different (`rename` does not support cross-buckets operations)"
            )

        self._client.copy_object(
            src_bucket=bucket,
            src_key=key,
            dst_bucket=new_bucket,
            dst_key=new_key,
        )
        logger.debug("rename: copied %s to %s successfully", path, new_path)
        self._delete_with_retry(bucket, key)
        logger.debug("rename: deleted %s successfully", path)

    def mkdir(self, path: Union[str, os.PathLike]) -> None:
        """No-op method for creating directories in S3 (not needed)."""
        pass

    def exists(self, path: Union[str, os.PathLike]) -> bool:
        logger.debug("exists: %s", path)

        bucket, key = parse_s3_uri(str(path))
        try:
            self._client.head_object(bucket, key)
        except S3Exception as e:
            if str(e) != "Service error: The object was not found":
                raise
            return False
        return True

    def rm_file(self, path: Union[str, os.PathLike]) -> None:
        logger.debug("rm_file: %s", path)

        bucket, key = parse_s3_uri(str(path))
        try:
            self._client.delete_object(bucket, key)
        except S3Exception:
            logger.exception("Failed to remove object from S3")

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        logger.debug("validate_checkpoint_id: %s", checkpoint_id)

        try:
            parse_s3_uri(str(checkpoint_id))
        except ValueError:
            return False
        return True

    @retry(
        retry=retry_if_exception_type(S3Exception),
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, max=5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.ERROR),
        reraise=True,
    )
    def _delete_with_retry(self, bucket_name: str, old_key: str):
        """Wrapper around :func:`S3Client.delete_object` to retry the deletion.

        Will retry a maximum of 3 times, only for `S3Exception`s, and wait between retries. It will reraise the caught
        exception too, and logs retries and final error, if any."""
        self._client.delete_object(bucket_name, old_key)


class S3StorageWriter(FileSystemWriter):
    def __init__(
        self,
        region: str,
        uri: str,
        **kwargs,
    ) -> None:
        """
        Initialize an S3 writer for distributed checkpointing.

        Args:
            region (str): The AWS region for S3.
            uri (str): The S3 URI to write checkpoints to.
            kwargs (dict): Keyword arguments to pass to the parent :class:`FileSystemWriter`.
        """
        super().__init__(
            path=uri,
            sync_files=False,  # FIXME: setting this to True makes the run to fail (L#333: `os.fsync(stream.fileno())`)
            **kwargs,
        )
        self.fs = S3FileSystem(region)  # type: ignore
        self.path = self.fs.init_path(uri)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return S3FileSystem.validate_checkpoint_id(checkpoint_id)


class S3StorageReader(FileSystemReader):
    def __init__(self, region: str, uri: str) -> None:
        """
        Initialize an S3 storage reader for distributed checkpointing.

        Args:
            region (str): The AWS region for S3.
            uri (str): The S3 URI to read checkpoints from.
        """
        super().__init__(path=uri)
        self.fs = S3FileSystem(region)  # type: ignore
        self.path = self.fs.init_path(uri)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return S3FileSystem.validate_checkpoint_id(checkpoint_id)
