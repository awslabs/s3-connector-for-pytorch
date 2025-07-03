#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import io
import logging
import os
import urllib.parse
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Union, Optional
from typing import List

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
import torch

from s3torchconnector._s3client import S3Client
from s3torchconnector._s3dataset_common import parse_s3_uri
from ..s3reader import S3ReaderConstructor, S3ReaderConstructorProtocol
from .. import S3ClientConfig
from .s3_prefix_strategy import S3PrefixStrategyBase, DefaultPrefixStrategy
from .._user_agent import UserAgent

logger = logging.getLogger(__name__)


class S3FileSystem(FileSystemBase):
    def __init__(
        self,
        region: str,
        s3_client: Optional[S3Client] = None,
        s3client_config: Optional[S3ClientConfig] = None,
        reader_constructor: Optional[S3ReaderConstructorProtocol] = None,
    ) -> None:
        """
        Initialize S3FileSystem.

        Args:
            region (str): The AWS region for S3.
            s3_client (Optional[S3Client]): Optional S3Client instance.
            s3client_config (Optional[S3ClientConfig]): Optional S3ClientConfig with parameters for S3 client.
            reader_constructor (Optional[S3ReaderConstructorProtocol]): Optional partial(S3Reader) created using S3ReaderConstructor
                e.g. S3ReaderConstructor.sequential() or S3ReaderConstructor.range_based()
        """
        self._path: Union[str, os.PathLike] = ""
        self._reader_constructor = reader_constructor or S3ReaderConstructor.default()

        # Get reader type string for user agent
        reader_type_string = S3ReaderConstructor.get_reader_type_string(
            self._reader_constructor
        )
        user_agent = UserAgent(
            ["dcp", torch.__version__, f"md/reader_type#{reader_type_string}"]
        )

        self._client = (
            S3Client(
                region=region, user_agent=user_agent, s3client_config=s3client_config
            )
            if s3_client is None
            else s3_client
        )

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
        path_str = _path_or_str_to_str(path)
        bucket, key = parse_s3_uri(path_str)

        if mode == "wb":  # write mode
            logger.debug("create_stream writable for %s", path_str)
            with self._client.put_object(bucket, key) as stream:
                yield stream
        elif mode == "rb":  # read mode
            logger.debug("create_stream readable for %s", path_str)
            with self._client.get_object(
                bucket, key, reader_constructor=self._reader_constructor
            ) as stream:
                yield stream
        else:
            raise ValueError(
                f"Invalid {mode=} mode argument: create_stream only supports rb (read mode) & wb (write mode)"
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
        self._path = path
        return self._path

    def rename(
        self, old_path: Union[str, os.PathLike], new_path: Union[str, os.PathLike]
    ) -> None:
        """Rename an object in S3.

        This is emulated by copying it to a new path and deleting the old path. The deletion part is retried (see also
        :func:`S3FileSystem._delete_with_retry`).

        Args:
            old_path (Union[str, os.PathLike]): The current path of the object.
            new_path (Union[str, os.PathLike]): The new path for the object.

        Raises:
            ValueError: If the old and new paths point to different buckets.
            S3Exception: If there is an error with the S3 client.
        """
        logger.debug("rename %s to %s", old_path, new_path)

        old_path_str = _path_or_str_to_str(old_path)
        new_path_str = _path_or_str_to_str(new_path)

        old_bucket, old_key = parse_s3_uri(old_path_str)
        escaped_old_key = self._escape_path(old_key)
        logger.debug("rename: escaped version of the source key: %s", escaped_old_key)
        new_bucket, new_key = parse_s3_uri(new_path_str)

        if old_bucket != new_bucket:
            raise ValueError(
                f"Source and destination buckets cannot be different (rename does not support cross-buckets operations)"
            )

        self._client.copy_object(
            src_bucket=old_bucket,
            src_key=escaped_old_key,
            dst_bucket=new_bucket,
            dst_key=new_key,
        )
        logger.debug("rename: copied %s to %s successfully", old_path_str, new_path_str)
        self._delete_with_retry(old_bucket, old_key)
        logger.debug("rename: s3://%s/%s successfully", old_bucket, old_key)

    def mkdir(self, path: Union[str, os.PathLike]) -> None:
        """No-op method for creating directories in S3 (not needed)."""
        pass

    def exists(self, path: Union[str, os.PathLike]) -> bool:
        logger.debug("exists %s", path)

        path_str = _path_or_str_to_str(path)
        bucket, key = parse_s3_uri(path_str)
        try:
            self._client.head_object(bucket, key)
        except S3Exception as e:
            if str(e) != "Service error: The object was not found":
                raise
            return False
        return True

    def rm_file(self, path: Union[str, os.PathLike]) -> None:
        logger.debug("remove %s", path)

        path_str = _path_or_str_to_str(path)
        bucket, key = parse_s3_uri(path_str)
        try:
            self._client.delete_object(bucket, key)
        except S3Exception:
            logger.exception("Failed to remove object from S3")

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        logger.debug("validate_checkpoint_id for %s", checkpoint_id)

        if isinstance(checkpoint_id, Path):
            return True

        try:
            parse_s3_uri(_path_or_str_to_str(checkpoint_id))
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

    @staticmethod
    def _escape_path(string):
        """URL-encodes path segments while preserving '/' separators using urllib.parse.quote().

        Args:
            string (str): URL path string to escape

        Returns:
            str: Path string with each segment percent-encoded, separators preserved
        """
        if not string:
            return string
        parts = []
        for part in string.split("/"):
            parts.append(urllib.parse.quote(part, safe=""))
        return "/".join(parts)


from torch.distributed.checkpoint.planner import SavePlan
import dataclasses
from dataclasses import dataclass


@dataclass
class StorageMetadata:
    """Metadata for S3 storage prefix."""

    prefix: str


class S3StorageWriter(FileSystemWriter):
    def __init__(
        self,
        region: str,
        path: str,
        s3client_config: Optional[S3ClientConfig] = None,
        prefix_strategy: Optional[S3PrefixStrategyBase] = None,
        **kwargs,
    ) -> None:
        """
        Initialize an S3 writer for distributed checkpointing.

        Args:
            region (str): The AWS region for S3.
            path (str): The S3 URI to write checkpoints to.
            s3client_config (Optional[S3ClientConfig]): Optional S3ClientConfig with parameters for S3 client.
            prefix_strategy (Optional[S3PrefixStrategyBase]): Optional strategy for generating S3 prefixes to
                optimize checkpoint organization and prevent throttling.
            kwargs (dict): Keyword arguments to pass to the parent :class:`FileSystemWriter`.
        """
        super().__init__(
            path=path,
            sync_files=False,  # FIXME: setting this to True makes the run to fail (L#333: `os.fsync(stream.fileno())`)
            **kwargs,
        )
        self.fs = S3FileSystem(region, s3client_config=s3client_config)  # type: ignore
        self.path = self.fs.init_path(path)
        self.prefix_strategy = prefix_strategy or DefaultPrefixStrategy()

    def prepare_global_plan(self, plans: List[SavePlan]) -> List[SavePlan]:
        """
        Prepare save plans with S3-specific storage metadata.

        Args:
            plans: List of save plans to be processed.

        Returns:
            Modified save plans with S3 storage metadata.
        """
        return [
            dataclasses.replace(
                plan, storage_data=StorageMetadata(self.prefix_strategy(idx))
            )
            for idx, plan in enumerate(plans)
        ]

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return S3FileSystem.validate_checkpoint_id(checkpoint_id)


class S3StorageReader(FileSystemReader):
    def __init__(
        self,
        region: str,
        path: Union[str, os.PathLike],
        s3client_config: Optional[S3ClientConfig] = None,
        reader_constructor: Optional[S3ReaderConstructorProtocol] = None,
    ) -> None:
        """
        Initialize an S3 reader for distributed checkpointing.

        Args:
            region (str): The AWS region for S3.
            path (Union[str, os.PathLike]): The S3 path to read checkpoints from.
            s3client_config (Optional[S3ClientConfig]): Optional S3ClientConfig with parameters for S3 client.
            reader_constructor (Optional[S3ReaderConstructorProtocol]): Optional partial(S3Reader) created using S3ReaderConstructor
                e.g. S3ReaderConstructor.sequential() or S3ReaderConstructor.range_based()
        """
        super().__init__(path)
        self.fs = S3FileSystem(region, s3client_config=s3client_config, reader_constructor=reader_constructor)  # type: ignore
        self.path = self.fs.init_path(path)
        self.sync_files = False

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return S3FileSystem.validate_checkpoint_id(checkpoint_id)


def _path_or_str_to_str(path: Union[str, os.PathLike]) -> str:
    return path if isinstance(path, str) else str(path)
