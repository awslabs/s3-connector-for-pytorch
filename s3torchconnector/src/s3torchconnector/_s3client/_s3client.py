#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import logging
import os
import gc
import threading
from functools import partial
from typing import Optional, Any

from s3torchconnector import S3Reader, S3Writer
from .s3client_config import S3ClientConfig

from s3torchconnectorclient._mountpoint_s3_client import (
    MountpointS3Client,
    ObjectInfo,
    HeadObjectResult,
    ListObjectStream,
    GetObjectStream,
    join_all_managed_threads,
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
NATIVE_S3_CLIENT = None


def _before_fork_handler():
    """Handler that cleans up CRT resources before fork operations."""
    global NATIVE_S3_CLIENT
    try:
        if NATIVE_S3_CLIENT is not None:
            # Release the client before fork as it's not fork-safe
            NATIVE_S3_CLIENT = None
            gc.collect()
            # Wait for native background threads to complete joining (0.5 sec timeout)
            join_all_managed_threads(0.5)
    except Exception as e:
        print(
            "Warning: Failed to properly clean up native background threads before fork. "
            f"Error: {e}\n"
            "Your subprocess may crash or hang. To prevent this:\n"
            "1. Ensure no active S3 client usage during fork operations\n"
            "2. Use multiprocessing with 'spawn' or 'forkserver' start method instead"
        )


# register the handler to release the S3 client and wait for background threads to join before fork happens
# As fork will not inherit any background threads. Wait for them to join to avoid crashes or hangs.
os.register_at_fork(before=_before_fork_handler)


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
        user_agent = user_agent or UserAgent()
        self._user_agent_prefix = user_agent.prefix
        self._s3client_config = s3client_config or S3ClientConfig()
        self._client_pid: Optional[int] = None
        global NATIVE_S3_CLIENT
        NATIVE_S3_CLIENT = None

    @property
    def _client(self) -> MountpointS3Client:
        global NATIVE_S3_CLIENT
        if (
            self._client_pid is None
            or self._client_pid != os.getpid()
            or NATIVE_S3_CLIENT is None
        ):
            # Acquire the lock to ensure thread-safety when creating the client.
            with _client_lock:
                if (
                    self._client_pid is None
                    or self._client_pid != os.getpid()
                    or NATIVE_S3_CLIENT is None
                ):
                    # This double-check ensures that the client is only created once.
                    NATIVE_S3_CLIENT = self._client_builder()
                    self._client_pid = os.getpid()

        assert NATIVE_S3_CLIENT is not None
        return NATIVE_S3_CLIENT

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
            max_attempts=self._s3client_config.max_attempts,
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
    def head_object(self, bucket: str, key: str) -> HeadObjectResult:
        log.debug(f"HeadObject s3://{bucket}/{key}")
        return self._client.head_object(bucket, key)

    def delete_object(self, bucket: str, key: str) -> None:
        log.debug(f"DeleteObject s3://{bucket}/{key}")
        self._client.delete_object(bucket, key)

    def copy_object(
        self, src_bucket: str, src_key: str, dst_bucket: str, dst_key: str
    ) -> None:
        log.debug(
            f"CopyObject s3://{src_bucket}/{src_key} to s3://{dst_bucket}/{dst_key}"
        )
        return self._client.copy_object(src_bucket, src_key, dst_bucket, dst_key)
