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


class RangedS3Reader(S3Reader):
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
    def bucket(self) -> str:
        return self._bucket

    @property
    def key(self) -> str:
        return self._key

    @cached_property
    def _object_info(self):
        return self._get_object_info()

    def _read_into_view(
        self, view: memoryview, start: int, end: int, buf_size: int
    ) -> int:
        """Creates a range-based stream and reads bytes from S3 into a memoryview.

        Args:
            view: Target memoryview to write data into
            start: Starting byte position in S3 object (inclusive)
            end: Ending byte position in S3 object (non inclusive)
            buf_size: Size of the buffer to fill

        Returns:
            int: Number of bytes read
        """
        # Create new stream for each read
        stream = self._get_stream(start, end)

        bytes_read = 0
        for chunk in stream:
            # Safeguard for buffer overflow (stream size > buf_size)
            chunk_size = min(len(chunk), buf_size - bytes_read)
            view[bytes_read : bytes_read + chunk_size] = chunk[:chunk_size]
            bytes_read += chunk_size
            # Exit if finished reading
            if bytes_read == buf_size:
                break

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

        return self._read_into_view(view, start, end, buf_size)

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

        self._read_into_view(view, start, end, byte_size)
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
