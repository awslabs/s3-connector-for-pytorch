#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from s3torchconnectorclient._mountpoint_s3_client import (
    MockMountpointS3Client,
    MountpointS3Client,
)

from . import S3Client

"""
_mock_s3client.py
    Internal client wrapper mock class for unit testing.
"""


class MockS3Client(S3Client):
    def __init__(self, region: str, bucket: str, part_size: int = 8 * 1024 * 1024):
        super().__init__(region)
        self._mock_client = MockMountpointS3Client(region, bucket, part_size=part_size)

    def add_object(self, key: str, data: bytes) -> None:
        self._mock_client.add_object(key, data)

    def remove_object(self, key: str) -> None:
        self._mock_client.remove_object(key)

    def _client_builder(self) -> MountpointS3Client:
        return self._mock_client.create_mocked_client()
