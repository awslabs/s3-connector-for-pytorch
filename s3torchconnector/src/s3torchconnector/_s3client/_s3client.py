#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import logging
import os
from functools import partial
from typing import Optional, Any

from s3torchconnector import S3Reader, S3Writer
from s3torchconnector._version import user_agent_prefix

from s3torchconnectorclient._mountpoint_s3_client import (
    MountpointS3Client,
    ObjectInfo,
    ListObjectStream,
    GetObjectStream,
)


"""
_s3client.py
    Internal client wrapper class on top of S3 client implementation 
    with multi-process support.
"""


log = logging.getLogger(__name__)


def _identity(obj: Any) -> Any:
    return obj


class S3Client:
    def __init__(self, region: str):
        self._region = region
        self._real_client = None
        self._client_pid = None

    @property
    def _client(self) -> MountpointS3Client:
        if self._client_pid is None or self._client_pid != os.getpid():
            self._client_pid = os.getpid()
            # `MountpointS3Client` does not survive forking, so re-create it if the PID has changed.
            self._real_client = self._client_builder()
        return self._real_client

    @property
    def region(self) -> str:
        return self._region

    def _client_builder(self) -> MountpointS3Client:
        return MountpointS3Client(
            region=self._region, user_agent_prefix=user_agent_prefix
        )

    def get_object(self, bucket: str, key: str) -> S3Reader:
        log.debug(f"GetObject s3://{bucket}/{key}")
        return S3Reader(
            bucket,
            key,
            get_object_info=partial(self.head_object, bucket, key),
            get_stream=partial(self._get_object_stream, bucket, key),
        )

    def _get_object_stream(self, bucket: str, key: str) -> GetObjectStream:
        return self._client.get_object(bucket, key)

    def put_object(
        self, bucket: str, key: str, storage_class: Optional[str] = None
    ) -> S3Writer:
        log.debug(f"PutObject s3://{bucket}/{key}")
        return S3Writer(self._client.put_object(bucket, key, storage_class))

    # TODO: Probably need a ListObjectResult on dataset side
    def list_objects(
        self, bucket: str, prefix: str = "", delimiter: str = "", max_keys: int = 1000
    ) -> ListObjectStream:
        log.debug(f"ListObjects s3://{bucket}/{prefix}")
        return self._client.list_objects(bucket, prefix, delimiter, max_keys)

    # TODO: We need ObjectInfo on dataset side
    def head_object(self, bucket: str, key: str) -> ObjectInfo:
        log.debug(f"HeadObject s3://{bucket}/{key}")
        return self._client.head_object(bucket, key)

    def delete_object(self, bucket: str, key: str) -> None:
        self._client.delete_object(bucket, key)

    def from_bucket_and_object_info(
        self, bucket: str, object_info: ObjectInfo
    ) -> S3Reader:
        log.debug(f"GetObjectWithInfo s3://{bucket}/{object_info.key}")
        return S3Reader(
            bucket,
            object_info.key,
            get_object_info=partial(_identity, object_info),
            get_stream=partial(self._get_object_stream, bucket, object_info.key),
        )
