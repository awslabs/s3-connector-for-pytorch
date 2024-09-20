import io
import os
from contextlib import contextmanager
from typing import Generator, Union
from torch.distributed.checkpoint.filesystem import FileSystem
from torch.distributed.checkpoint.filesystem import FileSystemWriter
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.filesystem import FileSystemBase

import boto3
from botocore.exceptions import ClientError

from s3torchconnector import S3Checkpoint
from s3torchconnector._s3dataset_common import parse_s3_uri  # type: ignore


class S3FS(FileSystemBase):
    def __init__(self, region: str) -> None:
        self.path = None
        self.region = region
        self.s3 = boto3.client("s3", region_name=region)
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
            io.IOBase: A stream for reading or writing to S3.

        Raises:
            ValueError: If the mode is not 'rb' or 'wb'.
        """
        if mode == "wb":  # write mode
            print(f"create_stream writable for {path}")
            with self.checkpoint.writer(path) as stream:
                yield stream
        elif mode == "rb":  # read mode
            print(f"create_stream readable for {path}")
            with self.checkpoint.reader(path) as stream:
                yield stream
        else:
            raise ValueError(
                "Invalid mode argument, create_stream only supports rb (read mode) & wb (write mode)"
            )

    def concat_path(
        self, path: Union[str, os.PathLike], suffix: str
    ) -> Union[str, os.PathLike]:
        """
        Concatenate a suffix to the given path.

        Args:
            path (Union[str, os.PathLike]): The base path.
            suffix (str): The suffix to concatenate.

        Returns:
            Union[str, os.PathLike]: The concatenated path.
        """
        print(f"concat_path for {path} and {suffix}")
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
        print(f"init_path for {path}")
        self.path = path
        return self.path

    def rename(
        self, path: Union[str, os.PathLike], new_path: Union[str, os.PathLike]
    ) -> None:
        """
        Rename an object in S3 by copying it to a new path and deleting the old path.

        Args:
            path (Union[str, os.PathLike]): The current path of the object.
            new_path (Union[str, os.PathLike]): The new path for the object.

        Raises:
            ClientError: If there is an error with the S3 client.
        """
        print(f"rename {path} to {new_path}")
        bucket_name, old_key = parse_s3_uri(path)
        _, new_key = parse_s3_uri(new_path)

        copy_source = {"Bucket": bucket_name, "Key": old_key}
        try:
            self.s3.copy_object(Bucket=bucket_name, CopySource=copy_source, Key=new_key)
            self.s3.delete_object(Bucket=bucket_name, Key=old_key)
        except ClientError as e:
            print(f"Error renaming object in S3: {e}")
            raise e

    def mkdir(self, path: [str, os.PathLike]) -> None:
        """
        No-op method for creating directories in S3 (not needed).
        """
        pass

    def exists(self, path: Union[str, os.PathLike]) -> bool:
        pass

    def rm_file(self, path: Union[str, os.PathLike]) -> None:
        pass

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        """
        Validate the checkpoint ID.

        Args:
            checkpoint_id (Union[str, os.PathLike]): The checkpoint ID to validate.

        Returns:
            bool: True if the checkpoint ID is valid, False otherwise.
        """
        print(f"validate_checkpoint_id for {checkpoint_id}")
        return FileSystem.validate_checkpoint_id(checkpoint_id)


class S3DPWriter(FileSystemWriter):
    def __init__(
        self,
        region: str,
        path: Union[str, os.PathLike],
        single_file_per_rank: bool = True,
        thread_count: int = 1,
        per_thread_copy_ahead: int = 10_000_000,
    ) -> None:
        """
        Initialize an S3 writer for distributed checkpointing.

        Args:
            region (str): The AWS region for S3.
            path (Union[str, os.PathLike]): The S3 path to write checkpoints.
            single_file_per_rank (bool, optional): Whether to write a single file per rank. Defaults to True.
            thread_count (int, optional): The number of threads to use for writing. Defaults to 1.
            per_thread_copy_ahead (int, optional): The number of bytes to copy ahead per thread. Defaults to 10_000_000.
        """
        super().__init__(
            path, single_file_per_rank, False, thread_count, per_thread_copy_ahead
        )
        self.fs = S3FS(region=region)
        self.path = self.fs.init_path(path)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return S3FS.validate_checkpoint_id(checkpoint_id)


class S3DPReader(FileSystemReader):
    def __init__(self, region: str, path: Union[str, os.PathLike]) -> None:
        """
        Initialize an S3 reader for distributed checkpointing.

        Args:
            region (str): The AWS region for S3.
            path (Union[str, os.PathLike]): The S3 path to read checkpoints from.
        """
        super().__init__(path)
        self.fs = S3FS(region=region)
        self.path = self.fs.init_path(path)
        self.sync_files = False

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return S3FS.validate_checkpoint_id(checkpoint_id)
