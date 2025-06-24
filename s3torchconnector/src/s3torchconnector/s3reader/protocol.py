#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from typing import Protocol, Callable, Optional, Union
from .s3reader import S3Reader
from s3torchconnectorclient._mountpoint_s3_client import (
    ObjectInfo,
    GetObjectStream,
    HeadObjectResult,
)


class GetStreamCallable(Protocol):
    def __call__(
        self, start: Optional[int] = None, end: Optional[int] = None
    ) -> GetObjectStream: ...


class S3ReaderConstructorProtocol(Protocol):
    def __call__(
        self,
        bucket: str,
        key: str,
        get_object_info: Callable[[], Union[ObjectInfo, HeadObjectResult]],
        get_stream: GetStreamCallable,
    ) -> S3Reader: ...
