from __future__ import annotations

import io
from typing import Callable, Iterator

from s3dataset._s3dataset import ObjectInfo

"""
s3_object.py
    File like representation of an S3 object.
"""


def _default_callback(s3_object: S3Object, data: bytes) -> None:
    return None


class S3Object(io.BufferedIOBase):
    def __init__(
        self,
        bucket: str,
        key: str,
        object_info: ObjectInfo = None,
        get_stream: Callable[[], Iterator[bytes]] = None,
        stream_callback: Callable[[S3Object, bytes], None] = None,
    ):
        if not bucket:
            raise ValueError("Bucket should be specified")
        self.bucket = bucket
        self.key = key
        self.object_info = object_info
        self._get_stream = get_stream
        self._stream = None
        self._stream_callback = stream_callback or _default_callback

    def prefetch(self):
        if self._stream is None:
            self._stream = self._wrap_stream(self._get_stream())

    def _wrap_stream(self, stream: Iterator[bytes]) -> Iterator[bytes]:
        for data in stream:
            self._stream_callback(self, data)
            yield data

    # TODO: Support multiple sizes
    def read(self, size=-1):
        if size != -1:
            raise NotImplementedError()
        self.prefetch()
        return b"".join(self._stream)
