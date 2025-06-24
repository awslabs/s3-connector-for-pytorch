#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import logging
import os
import gc
import threading
import traceback
import weakref
from functools import partial
from typing import Optional, Any, List

from s3torchconnector import S3Reader, S3Writer, S3ReaderConstructor
from s3torchconnector.s3reader.protocol import S3ReaderConstructorProtocol
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
_active_clients: "weakref.WeakSet[Any]" = weakref.WeakSet()


def _before_fork_handler():
    # Handler that cleans up CRT resources before fork operations.
    try:
        # Get all instances of S3Client that exist in the current process
        clients_list = _get_active_s3clients()
        if not clients_list:
            return

        for client in clients_list:
            if client._native_client is not None:
                # Release the client before fork as it's not fork-safe
                client._native_client = None
        # Clear the list of active clients. We will re-populate it after fork with only active clients.
        _reset_active_s3clients()
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
        traceback.print_exc()


def _after_fork_handler():
    """Handler that initializes the list of active clients after fork in child process."""
    _reset_active_s3clients()


# register the handler to release the S3 client and wait for background threads to join before fork happens
# As fork will not inherit any background threads. Wait for them to join to avoid crashes or hangs.
os.register_at_fork(before=_before_fork_handler, after_in_child=_after_fork_handler)


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
        self._native_client: Optional[MountpointS3Client] = None

    @property
    def _client(self) -> MountpointS3Client:
        if (
            self._client_pid is None
            or self._client_pid != os.getpid()
            or self._native_client is None
        ):
            # Acquire the lock to ensure thread-safety when creating the client.
            with _client_lock:
                if (
                    self._client_pid is None
                    or self._client_pid != os.getpid()
                    or self._native_client is None
                ):
                    # This double-check ensures that the client is only created once.
                    self._native_client = self._client_builder()
                    self._client_pid = os.getpid()
                    # Track the client in the list of active clients.
                    _active_clients.add(self)

        assert self._native_client is not None
        return self._native_client

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
            profile=self._s3client_config.profile,
            user_agent_prefix=self._user_agent_prefix,
            throughput_target_gbps=self._s3client_config.throughput_target_gbps,
            part_size=self._s3client_config.part_size,
            unsigned=self._s3client_config.unsigned,
            force_path_style=self._s3client_config.force_path_style,
            max_attempts=self._s3client_config.max_attempts,
        )

    def get_object(
        self,
        bucket: str,
        key: str,
        *,
        object_info: Optional[ObjectInfo] = None,
        reader_constructor: Optional[S3ReaderConstructorProtocol] = None,
    ) -> S3Reader:
        log.debug(f"GetObject s3://{bucket}/{key}, {object_info is None=}")
        if object_info is None:
            get_object_info = partial(self.head_object, bucket, key)
        else:
            get_object_info = partial(_identity, object_info)

        reader_constructor = reader_constructor or S3ReaderConstructor.default()

        return reader_constructor(
            bucket=bucket,
            key=key,
            get_object_info=get_object_info,
            get_stream=partial(self._get_object_stream, bucket, key),
        )

    def _get_object_stream(
        self,
        bucket: str,
        key: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> GetObjectStream:
        return self._client.get_object(bucket, key, start, end)

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


def _get_active_s3clients() -> List[S3Client]:
    """
    Returns a copy of list of all active S3 clients.
    Pay attention, it grabs a lock on _client_lock.

    Returns:
        List[S3Client]: A list of active S3 client instances.
    """
    with _client_lock:
        return list(_active_clients)


def _reset_active_s3clients():
    """
    Resets the list of active S3 clients.
    Pay attention, it grabs a lock on _client_lock.
    """
    global _active_clients
    with _client_lock:
        _active_clients = weakref.WeakSet()
