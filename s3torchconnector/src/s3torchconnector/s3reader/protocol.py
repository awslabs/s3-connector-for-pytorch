#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import os
from typing import (
    TYPE_CHECKING,
    Protocol,
    Callable,
    Optional,
    Union,
    List,
    Dict,
    runtime_checkable,
)
from .s3reader import S3Reader
from .dcp_optimized import ItemRange
from s3torchconnectorclient._mountpoint_s3_client import (
    ObjectInfo,
    GetObjectStream,
    HeadObjectResult,
)

if TYPE_CHECKING:
    from torch.distributed.checkpoint.planner import ReadItem
    from torch.distributed.checkpoint.metadata import MetadataIndex
    from torch.distributed.checkpoint.filesystem import _StorageInfo


class GetStreamCallable(Protocol):
    def __call__(
        self, start: Optional[int] = None, end: Optional[int] = None
    ) -> GetObjectStream: ...


@runtime_checkable
class S3ReaderConstructorProtocol(Protocol):
    def __call__(
        self,
        bucket: str,
        key: str,
        get_object_info: Callable[[], Union[ObjectInfo, HeadObjectResult]],
        get_stream: GetStreamCallable,
    ) -> S3Reader: ...


@runtime_checkable
class DCPS3ReaderConstructorProtocol(Protocol):
    _item_ranges_by_file: Dict[str, List[ItemRange]]

    def __call__(
        self,
        bucket: str,
        key: str,
        get_object_info: Callable[[], Union[ObjectInfo, HeadObjectResult]],
        get_stream: GetStreamCallable,
    ) -> S3Reader: ...

    def set_item_ranges_by_file(
        self,
        plan_items: "List[ReadItem]",
        storage_data: "Dict[MetadataIndex, _StorageInfo]",
        base_path: Union[str, os.PathLike],
    ) -> None: ...
