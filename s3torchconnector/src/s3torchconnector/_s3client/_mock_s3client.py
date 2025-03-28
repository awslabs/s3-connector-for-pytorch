#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from typing import Optional

from s3torchconnectorclient._mountpoint_s3_client import (
    MockMountpointS3Client,
    MountpointS3Client,
)

from . import S3Client
from .._user_agent import UserAgent
from .s3client_config import S3ClientConfig

"""
_mock_s3client.py
    Internal client wrapper mock class for unit testing.
"""


class MockS3Client(S3Client):
    def __init__(
        self,
        region: str,
        bucket: str,
        user_agent: Optional[UserAgent] = None,
        s3client_config: Optional[S3ClientConfig] = None,
    ):
        super().__init__(
            region,
            user_agent=user_agent,
            s3client_config=s3client_config,
        )
        self._mock_client = MockMountpointS3Client(
            region,
            bucket,
            throughput_target_gbps=self.s3client_config.throughput_target_gbps,
            part_size=self.s3client_config.part_size,
            user_agent_prefix=self.user_agent_prefix,
            unsigned=self.s3client_config.unsigned,
            force_path_style=self.s3client_config.force_path_style,
            max_attempts=self.s3client_config.max_attempts,
        )

    def add_object(self, key: str, data: bytes) -> None:
        self._mock_client.add_object(key, data)

    def remove_object(self, key: str) -> None:
        self._mock_client.remove_object(key)

    def _client_builder(self) -> MountpointS3Client:
        return self._mock_client.create_mocked_client()
