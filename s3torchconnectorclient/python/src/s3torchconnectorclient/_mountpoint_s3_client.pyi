#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from typing import List, Optional

# This interface is unstable!

class MountpointS3Client:
    throughput_target_gbps: float
    region: str
    part_size: int
    profile: Optional[str]
    unsigned: Optional[bool]
    force_path_style: Optional[bool]
    max_attempts: int
    user_agent_prefix: str
    endpoint: str

    def __init__(
        self,
        region: str,
        user_agent_prefix: str = "",
        throughput_target_gbps: float = 10.0,
        part_size: int = 8 * 1024 * 1024,
        profile: Optional[str] = None,
        unsigned: Optional[bool] = False,
        endpoint: Optional[str] = None,
        force_path_style: Optional[bool] = False,
        max_attempts: int = 10,
    ): ...
    def get_object(
        self,
        bucket: str,
        key: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> GetObjectStream: ...
    def put_object(
        self, bucket: str, key: str, storage_class: Optional[str] = None
    ) -> PutObjectStream: ...
    def list_objects(
        self, bucket: str, prefix: str = "", delimiter: str = "", max_keys: int = 1000
    ) -> ListObjectStream: ...
    def head_object(self, bucket: str, key: str) -> HeadObjectResult: ...
    def delete_object(self, bucket: str, key: str) -> None: ...
    def copy_object(
        self, src_bucket: str, src_key: str, dst_bucket: str, dst_key: str
    ) -> None: ...

class MockMountpointS3Client:
    throughput_target_gbps: float
    region: str
    part_size: int
    user_agent_prefix: str
    unsigned: bool
    force_path_style: bool
    max_attempts: int

    def __init__(
        self,
        region: str,
        bucket: str,
        endpoint: str = "",
        throughput_target_gbps: float = 10.0,
        part_size: int = 8 * 1024 * 1024,
        user_agent_prefix: str = "mock_client",
        unsigned: bool = False,
        force_path_style: bool = False,
        max_attempts: int = 10,
    ): ...
    def create_mocked_client(self) -> MountpointS3Client: ...
    def add_object(self, key: str, data: bytes) -> None: ...
    def remove_object(self, key: str) -> None: ...

class GetObjectStream:
    bucket: str
    key: str

    def __iter__(self) -> GetObjectStream: ...
    def __next__(self) -> bytes: ...
    def tell(self) -> int: ...

class PutObjectStream:
    bucket: str
    key: str
    def write(self, data: bytes) -> None: ...
    def close(self) -> None: ...

class RestoreStatus:
    in_progress: bool
    expiry: Optional[int]

    def __init__(self, in_progress: bool, expiry: Optional[int]): ...

class ObjectInfo:
    key: str
    etag: str
    size: int
    last_modified: int
    storage_class: Optional[str]
    restore_status: Optional[RestoreStatus]

    def __init__(
        self,
        key: str,
        etag: str,
        size: int,
        last_modified: int,
        storage_class: Optional[str],
        restore_status: Optional[RestoreStatus],
    ): ...

class HeadObjectResult:
    etag: str
    size: int
    last_modified: int
    storage_class: Optional[str]
    restore_status: Optional[RestoreStatus]

    def __init__(
        self,
        etag: str,
        size: int,
        last_modified: int,
        storage_class: Optional[str],
        restore_status: Optional[RestoreStatus],
    ): ...

class ListObjectResult:
    object_info: List[ObjectInfo]
    common_prefixes: List[str]

class ListObjectStream:
    bucket: str
    continuation_token: Optional[str]
    complete: bool
    prefix: str
    delimiter: str
    max_keys: int

    def __iter__(self) -> ListObjectStream: ...
    def __next__(self) -> ListObjectResult: ...
    @staticmethod
    def _from_state(
        client: MountpointS3Client,
        bucket: str,
        prefix: str,
        delimiter: str,
        max_keys: int,
        continuation_token: Optional[str],
        complete: bool,
    ) -> ListObjectStream: ...

class S3Exception(Exception):
    pass

__version__: str

def join_all_managed_threads(timeout_secs: float) -> None: ...
