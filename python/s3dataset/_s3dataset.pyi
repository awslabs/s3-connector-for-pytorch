from typing import List, Optional


# This interface is unstable!


class MountpointS3Client:
    region: str
    throughput_target_gbps: float
    part_size: int

    def __init__(self, region: str, throughput_target_gbps: float = 10.0, part_size: int = 8*1024*1024, profile: str = None, no_sign_request: bool = False): ...
    def get_object(self, bucket: str, key: str) -> GetObjectStream: ...
    def put_object(self, bucket: str, key: str) -> PutObjectStream: ...
    def list_objects(self, bucket: str, prefix: str = "", delimiter: str = "", max_keys: int = 1000) -> ListObjectStream: ...


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


class ObjectInfo:
    key: str
    etag: str
    size: int
    last_modified: int
    storage_class: Optional[str]
    restore_status: Optional[RestoreStatus]


class ListObjectResult:
    object_info: List[ObjectInfo]
    common_prefixes: List[str]


class ListObjectStream:
    bucket: str

    def __iter__(self) -> ListObjectStream: ...
    def __next__(self) -> ListObjectResult: ...
