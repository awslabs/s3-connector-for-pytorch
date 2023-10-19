import io
from s3dataset._s3dataset import MountpointS3Client, ObjectInfo, GetObjectStream


class S3Object(io.BufferedIOBase):
    def __init__(
        self,
        bucket: str,
        key: str,
        object_info: ObjectInfo = None,
        stream: GetObjectStream = None,
    ):
        if not bucket or not key:
            raise ValueError("Bucket and key should be specified")
        self.bucket = bucket
        self.key = key
        self.object_info = object_info
        self.stream = stream

    # TODO: Support multiple sizes
    def read(self, size=-1):
        return b"".join(self.stream)

