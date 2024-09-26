#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import logging
import os
import threading
from functools import partial
from typing import Optional, Any

from s3torchconnector import S3Reader, S3Writer
from .s3client_config import S3ClientConfig

from s3torchconnectorclient._mountpoint_s3_client import (
    MountpointS3Client,
    ObjectInfo,
    ListObjectStream,
    GetObjectStream,
)

from s3torchconnector._user_agent import UserAgent

"""
_s3client.py
    Internal client wrapper class on top of S3 client implementation 
    with multi-process support.
"""


log = logging.getLogger(__name__)


def _identity(obj: Any) -> Any:
    return obj


_client_lock = threading.Lock()


class S3Client:
    def __init__(
        self,
        region: str,
        *,
        endpoint: Optional[str] = None,
        user_agent: Optional[UserAgent] = None,
        s3client_config: Optional[S3ClientConfig] = None,
    ):
        self._region = region
        self._endpoint = endpoint
        self._real_client: Optional[MountpointS3Client] = None
        self._client_pid: Optional[int] = None
        user_agent = user_agent or UserAgent()
        self._user_agent_prefix = user_agent.prefix
        self._s3client_config = s3client_config or S3ClientConfig()

    @property
    def _client(self) -> MountpointS3Client:
        # This is a fast check to avoid acquiring the lock unnecessarily.
        if self._client_pid is None or self._client_pid != os.getpid():
            # Acquire the lock to ensure thread-safety when creating the client.
            with _client_lock:
                # This double-check ensures that the client is only created once.
                if self._client_pid is None or self._client_pid != os.getpid():
                    # `MountpointS3Client` does not survive forking, so re-create it if the PID has changed.
                    self._real_client = self._client_builder()
                    self._client_pid = os.getpid()
        assert self._real_client is not None
        return self._real_client

    @property
    def region(self) -> str:
        return self._region

    @property
    def s3client_config(self) -> S3ClientConfig:
        return self._s3client_config

    @property
    def user_agent_prefix(self) -> str:
        return self._user_agent_prefix

    def _client_builder(self) -> MountpointS3Client:
        return MountpointS3Client(
            region=self._region,
            endpoint=self._endpoint,
            user_agent_prefix=self._user_agent_prefix,
            throughput_target_gbps=self._s3client_config.throughput_target_gbps,
            part_size=self._s3client_config.part_size,
            unsigned=self._s3client_config.unsigned,
            force_path_style=self._s3client_config.force_path_style,
        )

    def get_object(
        self, bucket: str, key: str, *, object_info: Optional[ObjectInfo] = None
    ) -> S3Reader:
        log.debug(f"GetObject s3://{bucket}/{key}, {object_info is None=}")
        if object_info is None:
            get_object_info = partial(self.head_object, bucket, key)
        else:
            get_object_info = partial(_identity, object_info)

        return S3Reader(
            bucket,
            key,
            get_object_info=get_object_info,
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
        log.debug(f"DeleteObject s3://{bucket}/{key}")
        self._client.delete_object(bucket, key)
