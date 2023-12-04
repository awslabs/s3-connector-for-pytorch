#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

class S3BucketKey(object):
    """Read-only information about object stored in S3."""

    def __init__(
            self,
            bucket: str,
            key: str,
    ):
        if not bucket:
            raise ValueError("Bucket should be specified")
        self._bucket = bucket
        self._key = key

    @property
    def bucket(self):
        return self._bucket

    @property
    def key(self):
        return self._key
