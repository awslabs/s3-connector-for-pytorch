import io
from typing import Callable

from s3dataset._s3dataset import ObjectInfo, GetObjectStream

"""
s3_object.py
    File like representation of an S3 object.
"""


class S3Object(io.BufferedIOBase):
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

    def prefetch(self):
        if self._stream is None:
            self._stream = self._get_stream()

    # TODO: Support multiple sizes
    def read(self, size=-1):
        if size != -1:
            raise NotImplementedError()
        self.prefetch()
        return b"".join(self._stream)
