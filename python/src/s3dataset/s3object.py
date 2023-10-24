import io
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
        stream: GetObjectStream = None,
    ):
        if not bucket:
            raise ValueError("Bucket should be specified")
        self.bucket = bucket
        self.key = key
        self.object_info = object_info
        self.stream = stream

    # TODO: Support multiple sizes
    def read(self, size=-1):
        if size != -1:
            raise NotImplementedError()
        return b"".join(self.stream)
