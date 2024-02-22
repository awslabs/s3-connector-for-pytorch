#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
from typing import NamedTuple, Optional

from s3torchconnectorclient._mountpoint_s3_client import ObjectInfo


class S3BucketKeyData(NamedTuple):
    """Read-only information about object stored in S3."""

    bucket: str
    key: str
    object_info: Optional[ObjectInfo] = None
