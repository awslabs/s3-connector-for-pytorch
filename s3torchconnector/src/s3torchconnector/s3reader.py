#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from __future__ import annotations

import io
from functools import cached_property
from io import SEEK_CUR, SEEK_END, SEEK_SET
from typing import Callable, Optional, Iterator, Union, cast
from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass

from s3torchconnectorclient._mountpoint_s3_client import (
    ObjectInfo,
    GetObjectStream,
    HeadObjectResult,
)


@dataclass(frozen=True)
class S3ReaderConfig:
    """A dataclass exposing configurable parameters for S3Reader.

    Args:
    reader_type (ReaderType): Determines the S3 access strategy.
        SEQUENTIAL:  Buffers entire object sequentially. Best for full reads and repeated access.
        RANGE_BASED: Fetches specific byte ranges on-demand. Suitable for partial reads of large objects.
    """

    class ReaderType(Enum):
        SEQUENTIAL = "sequential"
        RANGE_BASED = "range"

    # Default to _SequentialS3Reader
    reader_type: ReaderType = ReaderType.SEQUENTIAL

    @classmethod
    def sequential(cls) -> S3ReaderConfig:
        """Alternative constructor for sequential reading configuration."""
        return cls(reader_type=cls.ReaderType.SEQUENTIAL)

    @classmethod
    def range_based(cls) -> S3ReaderConfig:
        """Alternative constructor for range-based reading configuration."""
        return cls(reader_type=cls.ReaderType.RANGE_BASED)


class _BaseS3Reader(ABC, io.BufferedIOBase):
    """Abstract base class for S3 reader implementations."""

    @property
    @abstractmethod
    def bucket(self) -> str:
        pass

    @property
    @abstractmethod
    def key(self) -> str:
        pass

    @abstractmethod
    def read(self, size: Optional[int] = None) -> bytes:
        pass

    @abstractmethod
    def seek(self, offset: int, whence: int = SEEK_SET, /) -> int:
        pass

    @abstractmethod
    def tell(self) -> int:
        pass

    @abstractmethod
    def readinto(self, buf) -> int:
        pass


class _SequentialS3Reader(_BaseS3Reader):
    """Sequential S3 reader implementation

    Maintains an internal buffer and reads data sequentially from S3.
    This implementation is optimal for:
    - Full sequential reads
    - Repeated access to the same data
    - Scenarios where data is typically read in order
    """

    def __init__(
        self,
        bucket: str,
        key: str,
        get_object_info: Callable[[], Union[ObjectInfo, HeadObjectResult]],
        get_stream: Callable[[], GetObjectStream],
    ):
        if not bucket:
            raise ValueError("Bucket should be specified")
        self._bucket = bucket
        self._key = key
        self._get_object_info = get_object_info
        self._get_stream = get_stream
        self._stream: Optional[Iterator[bytes]] = None
        self._buffer = io.BytesIO()
        self._size: Optional[int] = None
        # Invariant: _position == _buffer._tell() unless _position_at_end()
        self._position = 0

    @property
    def bucket(self):
        return self._bucket

    @property
    def key(self):
        return self._key

    @cached_property
    def _object_info(self):
        return self._get_object_info()

    def prefetch(self) -> None:
        """Start fetching data from S3.

        Raises:
            S3Exception: An error occurred accessing S3.
        """

        if self._stream is None:
            self._stream = self._get_stream()

    def readinto(self, buf) -> int:
        """Read up to len(buf) bytes into a pre-allocated, writable bytes-like object buf.
        Return the number of bytes read. If no bytes are available, zero is returned.

        Args:
            buf : writable bytes-like object

        Returns:
            int : numer of bytes read or zero, if no bytes available
        """
        buf_size = len(buf)
        if self._position_at_end() or buf_size == 0:
            # If no bytes are available or no place to write data, zero should be returned
            return 0

        self.prefetch()
        assert self._stream is not None

        cur_pos = self._position
        # preload enough bytes in buffer
        self.seek(buf_size, SEEK_CUR)
        # restore position, before starting to write into buf
        self._buffer.seek(cur_pos)
        size = self._buffer.readinto(buf)
        self._position = self._buffer.tell()

        return size

    def read(self, size: Optional[int] = None) -> bytes:
        """Read up to size bytes from the object and return them.

        If size is zero or positive, read that many bytes from S3, or until the end of the object.
        If size is None or negative, read the entire file.

        Args:
            size (int | None): how many bytes to read.

        Returns:
            bytes: Bytes read from S3 Object

        Raises:
            S3Exception: An error occurred accessing S3.
        """

        if size is not None and not isinstance(size, int):
            raise TypeError(f"argument should be integer or None, not {type(size)!r}")
        if self._position_at_end():
            # Invariant: if we're at EOF, it doesn't matter what `size` is, we'll always return no data and have no
            # side effect.
            return b""

        self.prefetch()
        assert self._stream is not None
        cur_pos = self._position
        if size is None or size < 0:
            # Special case read() all to use O(n) algorithm
            self._buffer.seek(0, SEEK_END)
            for batch in self._stream:
                self._buffer.write(batch)

            # Once we've emptied the buffer, we'll always be at EOF!
            self._size = self._buffer.tell()
        else:
            self.seek(size, SEEK_CUR)

        self._buffer.seek(cur_pos)
        data = self._buffer.read(size)
        self._position = self._buffer.tell()
        return data

    def seek(self, offset: int, whence: int = SEEK_SET, /) -> int:
        """Change the stream position to the given byte offset, interpreted relative to whence.

        When seeking beyond the end of the file, always stay at EOF.
        Seeking before the start of the file results in a ValueError.

        Args:
            offset (int): How many bytes to seek relative to whence.
            whence (int): One of SEEK_SET, SEEK_CUR, and SEEK_END. Default: SEEK_SET

        Returns:
            int: Current position of the stream

        Raises:
            S3Exception: An error occurred accessing S3.

        """
        if not isinstance(offset, int):
            raise TypeError(f"integer argument expected, got {type(offset)!r}")
        if whence == SEEK_END:
            if offset >= 0:
                self._position = self._get_size()
                return self._position
            offset += self._get_size()
        elif whence == SEEK_CUR:
            if self._position_at_end() and offset >= 0:
                return self._position
            offset += self._position
        elif whence == SEEK_SET:
            pass
        elif isinstance(whence, int):
            raise ValueError("Seek must be passed SEEK_CUR, SEEK_SET, or SEEK_END")
        else:
            raise TypeError(f"integer argument expected, got {type(whence)!r}")

        if offset < 0:
            raise ValueError(f"negative seek value {offset}")

        if offset > self._buffer_size():
            self._prefetch_to_offset(offset)

        self._position = self._buffer.seek(offset)
        return self._position

    def _prefetch_to_offset(self, offset: int) -> None:
        self.prefetch()
        assert self._stream is not None
        buf_size = self._buffer.seek(0, SEEK_END)
        try:
            while offset > buf_size:
                buf_size += self._buffer.write(next(self._stream))
        except StopIteration:
            self._size = self._buffer.tell()

    def _get_size(self) -> int:
        if self._size is None:
            self._size = self._object_info.size
        return self._size

    def _position_at_end(self) -> bool:
        # Code calling this must only be used for optimisation purposes.
        if self._size is None:
            # We can never be special cased to EOF if we never saw how long it is.
            # If we _are_ at EOF, we'll just not take the early exits.
            return False
        return self._position == self._size

    def _buffer_size(self) -> int:
        cur_pos = self._buffer.tell()
        self._buffer.seek(0, SEEK_END)
        buffer_size = self._buffer.tell()
        self._buffer.seek(cur_pos)
        return buffer_size

    def tell(self) -> int:
        """
        Returns:
              int: Current stream position.
        """
        return self._position

    def readable(self) -> bool:
        """
        Returns:
            bool: Return whether object was opened for reading.
        """
        return True

    def writable(self) -> bool:
        """
        Returns:
            bool: Return whether object was opened for writing.
        """
        return False


class _RangedS3Reader(_BaseS3Reader):
    """Range-based S3 reader implementation

    Provides efficient random access to S3 objects by requesting specific byte ranges.

    This reader is optimal for:
    - Random access patterns
    - Partial reads of large objects
    - Memory-constrained scenarios where buffering full objects is impractical
    """

    def __init__(
        self,
        bucket: str,
        key: str,
        get_object_info: Callable[[], Union[ObjectInfo, HeadObjectResult]],
        get_stream: Callable[[Optional[int], Optional[int]], GetObjectStream],
    ):
        if not bucket:
            raise ValueError("Bucket should be specified")
        self._bucket = bucket
        self._key = key
        self._get_object_info = get_object_info
        self._get_stream = get_stream
        self._stream: Optional[Iterator[bytes]] = None
        self._size: Optional[int] = None
        self._position = 0

    @property
    def bucket(self):
        return self._bucket

    @property
    def key(self):
        return self._key

    @cached_property
    def _object_info(self):
        return self._get_object_info()

    def readinto(self, buf) -> int:
        """Read up to len(buf) bytes into a pre-allocated, writable bytes-like object buf.
        Return the number of bytes read. If no bytes are available, zero is returned.

        Args:
            buf : writable bytes-like object

        Returns:
            int : numer of bytes read or zero, if no bytes available
        """

        buf_size = len(buf)
        if self._position_at_end() or buf_size == 0:
            # If no bytes are available or no place to write data, zero should be returned
            return 0

        # Calculate the range to request
        start = self._position
        end = min(start + buf_size, self._get_size())

        # Return no data if zero-length range
        if start >= end:
            return 0

        # Create memoryview of the target buffer
        view = memoryview(buf)

        # Get stream for specified byte range
        self._stream = self._get_stream(start, end)

        bytes_read = 0
        for chunk in self._stream:
            chunk_size = min(len(chunk), buf_size - bytes_read)
            view[bytes_read : bytes_read + chunk_size] = chunk[:chunk_size]
            bytes_read += chunk_size
            if bytes_read == buf_size:
                break

        self._position += bytes_read
        return bytes_read

    def read(self, size: Optional[int] = None) -> bytes:
        """Read up to size bytes from the object and return them.

        If size is zero or positive, read that many bytes from S3, or until the end of the object.
        If size is None or negative, read the entire file.

        Args:
            size (int | None): how many bytes to read.

        Returns:
            bytes: Bytes read from S3 Object

        Raises:
            S3Exception: An error occurred accessing S3.
        """

        if size is not None and not isinstance(size, int):
            raise TypeError(f"argument should be integer or None, not {type(size)!r}")
        if self._position_at_end():
            # Invariant: if we're at EOF, it doesn't matter what `size` is, we'll always return no data and have no
            # side effect.
            return b""

        # Calculate the range to request
        start = self._position
        if size is None or size < 0:
            end = self._get_size()
        else:
            end = min(start + size, self._get_size())

        # Return no data if zero-length range
        if start >= end:
            return b""

        # Pre-allocate buffer
        byte_size = end - start
        buffer = bytearray(byte_size)
        view = memoryview(buffer)

        # Get stream for specified byte range
        self._stream = self._get_stream(start, end)

        bytes_read = 0
        for chunk in self._stream:
            chunk_size = min(len(chunk), byte_size - bytes_read)
            view[bytes_read : bytes_read + chunk_size] = chunk[:chunk_size]
            bytes_read += chunk_size
            if bytes_read >= byte_size:
                break

        self._position += bytes_read
        return view.tobytes()

    def seek(self, offset: int, whence: int = SEEK_SET, /) -> int:
        """Change the stream position to the given byte offset, interpreted relative to whence.

        When seeking beyond the end of the file, always stay at EOF.
        Seeking before the start of the file results in a ValueError.

        Args:
            offset (int): How many bytes to seek relative to whence.
            whence (int): One of SEEK_SET, SEEK_CUR, and SEEK_END. Default: SEEK_SET

        Returns:
            int: Current position of the stream

        Raises:
            S3Exception: An error occurred accessing S3.

        """
        if not isinstance(offset, int):
            raise TypeError(f"integer argument expected, got {type(offset)!r}")
        if whence == SEEK_END:
            if offset >= 0:
                self._position = self._get_size()
                return self._position
            offset += self._get_size()
        elif whence == SEEK_CUR:
            if self._position_at_end() and offset >= 0:
                return self._position
            offset += self._position
        elif whence == SEEK_SET:
            pass
        elif isinstance(whence, int):
            raise ValueError("Seek must be passed SEEK_CUR, SEEK_SET, or SEEK_END")
        else:
            raise TypeError(f"integer argument expected, got {type(whence)!r}")

        if offset < 0:
            raise ValueError(f"negative seek value {offset}")

        # Update position without prefetching.
        self._position = min(offset, self._get_size())
        return self._position

    def _get_size(self) -> int:
        if self._size is None:
            self._size = self._object_info.size
        return self._size

    def _position_at_end(self) -> bool:
        # Code calling this must only be used for optimisation purposes.
        if self._size is None:
            # We can never be special cased to EOF if we never saw how long it is.
            # If we _are_ at EOF, we'll just not take the early exits.
            return False
        return self._position == self._size

    def tell(self) -> int:
        """
        Returns:
              int: Current stream position.
        """
        return self._position

    def readable(self) -> bool:
        """
        Returns:
            bool: Return whether object was opened for reading.
        """
        return True

    def writable(self) -> bool:
        """
        Returns:
            bool: Return whether object was opened for writing.
        """
        return False


class S3Reader(io.BufferedIOBase):
    """A read-only, file like representation of a single object stored in S3.

    This class acts as a factory that returns either a sequential or range-based reader
    implementation based on the provided configuration.

    Args:
        bucket (str): S3 bucket name
        key (str): Object key in the bucket
        get_object_info: Callable that returns object metadata
        get_stream: Callable that returns object data stream
        reader_config (S3ReaderConfig, optional): Configuration for reader behavior.
    """

    def __new__(
        cls,
        bucket: str,
        key: str,
        get_object_info: Callable[[], Union[ObjectInfo, HeadObjectResult]],
        get_stream: Union[
            Callable[[], GetObjectStream],
            Callable[[Optional[int], Optional[int]], GetObjectStream],
        ],
        reader_config: Optional[S3ReaderConfig] = None,
    ):
        config = reader_config or S3ReaderConfig()

        if config.reader_type == S3ReaderConfig.ReaderType.SEQUENTIAL:
            return _SequentialS3Reader(
                bucket,
                key,
                get_object_info,
                cast(Callable[[], GetObjectStream], get_stream),
            )
        elif config.reader_type == S3ReaderConfig.ReaderType.RANGE_BASED:
            return _RangedS3Reader(
                bucket,
                key,
                get_object_info,
                cast(
                    Callable[[Optional[int], Optional[int]], GetObjectStream],
                    get_stream,
                ),
            )

        raise ValueError(f"Unsupported reader type: {config.reader_type}")
