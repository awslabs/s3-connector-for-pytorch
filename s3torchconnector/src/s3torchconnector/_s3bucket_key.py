#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
from typing import NamedTuple


class S3BucketKey(NamedTuple):
    """Read-only information about object stored in S3."""

    bucket: str
    key: str
