import os
from functools import partial
from typing import Optional

from s3dataset_s3_client._s3dataset import (
    MountpointS3Client,
    ObjectInfo,
    ListObjectStream,
    GetObjectStream,
)

from s3dataset_s3_client import S3Object
from s3dataset_s3_client.put_object_stream_wrapper import PutObjectStreamWrapper


class S3Client:
    def __init__(self, region: str):
        self._region = region
        self._real_client = None
        self._client_pid = None

    @property
    def _client(self) -> MountpointS3Client:
        if self._client_pid != os.getpid():
            self._client_pid = os.getpid()
            # `MountpointS3Client` does not survive forking, so re-create it if the PID has changed.
            self._real_client = MountpointS3Client(region=self._region)
        return self._real_client

    @property
    def region(self) -> str:
        return self._region

    # TODO: S3Object to become S3Reader
    def get_object(self, bucket: str, key: str) -> S3Object:
        return S3Object(bucket, key, get_stream=partial(self._get_object, bucket, key))

    def _get_object(self, bucket: str, key: str) -> GetObjectStream:
        return self._client.get_object(bucket, key)

    # TODO: PutObjectStreamWrapper -> S3Writer
    def put_object(
        self, bucket: str, key: str, storage_class: Optional[str] = None
    ) -> PutObjectStreamWrapper:
        return PutObjectStreamWrapper(
            self._client.put_object(bucket, key, storage_class)
        )

    # TODO: Probably need a ListObjectResult on dataset side
    def list_objects(
        self, bucket: str, prefix: str = "", delimiter: str = "", max_keys: int = 1000
    ) -> ListObjectStream:
        return self._client.list_objects(bucket, prefix, delimiter, max_keys)

    # TODO: We need ObjectInfo on dataset side
    def head_object(self, bucket: str, key: str) -> ObjectInfo:
        return self._client.head_object(bucket, key)
