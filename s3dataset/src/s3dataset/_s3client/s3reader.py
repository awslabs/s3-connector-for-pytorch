import io
from io import SEEK_CUR, SEEK_END, SEEK_SET
from typing import Callable, Optional

from s3dataset_s3_client._s3dataset import ObjectInfo, GetObjectStream

"""
s3reader.py
    File like representation of a readable S3 object.
"""


class S3Reader(io.BufferedIOBase):
    def __init__(
        self,
        bucket: str,
        key: str,
        object_info: ObjectInfo = None,
        get_stream: Callable[[], GetObjectStream] = None,
    ):
        if not bucket:
            raise ValueError("Bucket should be specified")
        self.bucket = bucket
        self.key = key
        self.object_info = object_info
        self._get_stream = get_stream
        self._stream = None
        self._buffer = io.BytesIO()
        if object_info is not None:
            self._size = object_info.size
        else:
            self._size = None
        # Invariant: _position == _buffer._tell() unless _position_at_end()
        self._position = 0

    def prefetch(self) -> None:
        """
        Start fetching data from S3.
        """
        if self._stream is None:
            self._stream = self._get_stream()

    def read(self, size: Optional[int] = None) -> bytes:
        """
        Returns the bytes read.
        Args:
            size: how many bytes to read.
        If size is zero or positive, read that many bytes from S3, or until the end of the object.
        If size is None or negative, read the entire file.
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
        """
        Returns the new cursor position
        Args:
            offset: How many bytes to seek relative to whence.
            whence: One of SEEK_SET, SEEK_CUR, and SEEK_END.
                    Default: SEEK_SET

        When seeking beyond the end of the file, always stay at EOF.
        Seeking before the start of the file results in a ValueError.
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
            raise NotImplementedError("TODO - implement HeadObject")
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
        return self._position

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False
