#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import io
from functools import cached_property
from io import SEEK_CUR, SEEK_END, SEEK_SET
from typing import Callable, Optional, Iterator, Union

from s3torchconnectorclient._mountpoint_s3_client import (
    ObjectInfo,
    GetObjectStream,
    HeadObjectResult,
)

from .s3reader import S3Reader

DEFAULT_BUFFER_SIZE = 8 * 1024 * 1024  # 8MB


class RangedS3Reader(S3Reader):
    """Range-based S3 reader implementation with adaptive buffering.

    Performs byte-range requests to read specific portions of S3 objects without
    downloading the entire file. Includes optional adaptive buffer to reduce S3 API
    calls for small, sequential reads while bypassing buffering for large reads.
    Optimal for sparse partial reads of large objects.

    Buffering behavior:

    * Small reads (< ``buffer_size``): Loads ``buffer_size`` bytes to buffer, copies to user
    * Large reads (>= ``buffer_size``): Direct S3 access, bypass buffer
    * Forward overlapping reads: Reuses existing buffer data if possible when read range extends beyond current buffer
    * Buffer can be disabled by setting ``buffer_size`` to 0
    * If ``buffer_size`` is None, uses default 8MB buffer

    Args:
        bucket: S3 bucket name
        key: S3 object key
        get_object_info: Callable that returns object metadata
        get_stream: Callable that returns stream for byte range requests
        buffer_size: Internal buffer size in bytes, defaults to 8MB
    """

    def __init__(
        self,
        bucket: str,
        key: str,
        get_object_info: Callable[[], Union[ObjectInfo, HeadObjectResult]],
        get_stream: Callable[[Optional[int], Optional[int]], GetObjectStream],
        buffer_size: Optional[int] = None,
    ):
        if not bucket:
            raise ValueError("Bucket should be specified")
        self._bucket = bucket
        self._key = key
        self._get_object_info = get_object_info
        self._get_stream = get_stream
        self._size: Optional[int] = None
        self._position: int = 0

        # Buffer Parameters
        self._buffer_size: int
        self._enable_buffering: bool
        if buffer_size is None:  # If None, use default buffer size
            self._buffer_size = DEFAULT_BUFFER_SIZE
            self._enable_buffering = True
        else:  # If integer, enable buffering if > 0 (0 disables buffer)
            self._buffer_size = buffer_size
            self._enable_buffering = buffer_size > 0
        # Create reusable buffer
        self._buffer: Optional[bytearray] = (
            bytearray(self._buffer_size) if self._enable_buffering else None
        )
        self._buffer_view: Optional[memoryview] = (
            memoryview(self._buffer) if self._buffer else None
        )
        # Track buffer byte range
        self._buffer_start: int = 0
        self._buffer_end: int = 0

    @property
    def bucket(self) -> str:
        return self._bucket

    @property
    def key(self) -> str:
        return self._key

    @cached_property
    def _object_info(self):
        return self._get_object_info()

    def _load_buffer(self, start: int):
        """Load self._buffer with ranged request

        Args:
            start: Starting byte position to load into buffer
        """
        end = min(start + self._buffer_size, self._get_size())

        assert self._buffer_view is not None  # for type checker
        bytes_read = 0
        # Reuse buffer
        for chunk in self._get_stream(start, end):
            chunk_len = len(chunk)
            self._buffer_view[bytes_read : bytes_read + chunk_len] = chunk
            bytes_read += chunk_len

        # Update Buffer Boundaries
        self._buffer_start, self._buffer_end = start, start + bytes_read

    def _read_buffered(self, view: memoryview, start: int, end: int) -> int:
        """Read data from internal buffer into the provided memoryview.
        Loads buffer if buffer doesn't contain full data range.

        Args:
            view: Target memoryview to write data into
            start: Starting byte position in S3 object (inclusive)
            end: Ending byte position in S3 object (exclusive)

        Returns:
            int: Number of bytes read
        """
        if start < self._buffer_start or end > self._buffer_end:
            self._load_buffer(start)

        buffer_offset = start - self._buffer_start
        length = end - start

        assert self._buffer is not None  # for type checker
        view[:length] = self._buffer[buffer_offset : buffer_offset + length]

        return length

    def _read_unbuffered(self, view: memoryview, start: int, end: int) -> int:
        """Creates a range-based stream and reads bytes from S3 into a memoryview.

        Args:
            view: Target memoryview to write data into
            start: Starting byte position in S3 object (inclusive)
            end: Ending byte position in S3 object (exclusive)

        Returns:
            int: Number of bytes read
        """
        length = end - start
        bytes_read = 0
        for chunk in self._get_stream(start, end):
            # Safeguard for buffer overflow (stream size > length)
            chunk_size = min(len(chunk), length - bytes_read)
            view[bytes_read : bytes_read + chunk_size] = chunk[:chunk_size]
            bytes_read += chunk_size
            # Exit if finished reading
            if bytes_read == length:
                break

        return bytes_read

    def _read_range(self, view: memoryview, start: int, end: int) -> int:
        """Reads into a memoryview from a specific byte range.
        Dispatch read request to buffered or unbuffered implementation based on request size.

        - Forward overlap optimization: When read range starts within current buffer and extends beyond it,
        first uses the overlapped portion from buffer, then processes remaining portion according to size.
        - Large (remaining) requests (< buffer_size): bypass buffering to speed up data transfer
        - Small (remaining) requests (>= buffer_size): use buffering to reduce S3 API calls, anticipating next call

        Args:
            view: Target memoryview to write data into
            start: Starting byte position in S3 object (inclusive)
            end: Ending byte position in S3 object (exclusive)

        Returns:
            int: Number of bytes read
        """

        bytes_read = 0

        # Forward overlap case: load from overlapped part in buffer first
        # Only apply when starting within buffer and extending beyond it, or exact match
        if self._buffer_start <= start < self._buffer_end <= end:
            # Read overlapped portion from buffer
            overlap_read = self._read_buffered(view, start, self._buffer_end)
            # Adjust start and view for the remaining portion
            start = self._buffer_end
            view = view[overlap_read:]
            bytes_read += overlap_read

        # Handle remaining portion / full range based on size
        if end - start >= self._buffer_size or not self._enable_buffering:
            # Large reads: return data directly
            bytes_read += self._read_unbuffered(view, start, end)
        else:
            # Small reads: buffer data and return from buffer
            bytes_read += self._read_buffered(view, start, end)

        self._position += bytes_read
        return bytes_read

    def readinto(self, buf) -> int:
        """Read up to len(buf) bytes into a pre-allocated, writable bytes-like object buf.
        Return the number of bytes read. If no bytes are available, zero is returned.

        Args:
            buf : writable bytes-like object

        Returns:
            int : numer of bytes read or zero, if no bytes available
        """

        try:
            view = memoryview(buf)
            if view.readonly:
                raise TypeError(
                    f"argument must be a writable bytes-like object, not {type(buf).__name__}"
                )
        except TypeError:
            raise TypeError(
                f"argument must be a writable bytes-like object, not {type(buf).__name__}"
            )

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

        return self._read_range(view, start, end)

    def read(self, size: Optional[int] = None) -> bytes:
        """Read up to size bytes from the current position.

        If size is zero or positive, read that many bytes from S3, or until the end of the object.
        If size is None or negative, read until the end of the object.

        Args:
            size (int | None): how many bytes to read.

        Returns:
            bytes: Bytes read from specified range.

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

        self._read_range(view, start, end)
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
