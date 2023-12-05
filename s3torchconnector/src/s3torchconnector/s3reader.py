#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import io
from functools import cached_property
from io import SEEK_CUR, SEEK_END, SEEK_SET
from typing import Callable, Optional

from s3torchconnectorclient._mountpoint_s3_client import ObjectInfo, GetObjectStream


class S3Reader(io.BufferedIOBase):
    """A read-only, file like representation of a single object stored in S3."""

    def __init__(
        self,
        bucket: str,
        key: str,
        get_object_info: Callable[[], ObjectInfo] = None,
        get_stream: Callable[[], GetObjectStream] = None,
    ):
        if not bucket:
            raise ValueError("Bucket should be specified")
        self._bucket = bucket
        self._key = key
        self._get_object_info = get_object_info
        self._get_stream = get_stream
        self._stream = None
        self._buffer = io.BytesIO()
        self._size = None
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
        cur_pos = self._position
        if size is None or size < 0:
            # Special case read() all to use O(n) algorithm
            self._buffer.seek(0, SEEK_END)
            self._buffer.write(b"".join(self._stream))
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
