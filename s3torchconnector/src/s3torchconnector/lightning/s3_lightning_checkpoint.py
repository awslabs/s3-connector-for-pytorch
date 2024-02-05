#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from typing import Optional, Dict, Any

import torch

from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.plugins.io import CheckpointIO

from s3torchconnector._s3client import S3Client
from s3torchconnector._s3dataset_common import parse_s3_uri


class S3LightningCheckpoint(CheckpointIO):
    def __init__(self, region: str):
        self.region = region
        self._client = S3Client(region)

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        s3_uri: _PATH,
        storage_options: Optional[Any] = None,
    ) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write."""
        bucket, key = parse_s3_uri(s3_uri)
        with self._client.put_object(bucket, key) as s3writer:
            torch.save(checkpoint, s3writer)

    def load_checkpoint(
        self, s3_uri: _PATH, map_location: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Load checkpoint from a path when resuming or loading ckpt for test/validate/predict stages."""
        bucket, key = parse_s3_uri(s3_uri)
        s3reader = self._client.get_object(bucket, key)
        return torch.load(s3reader, map_location)

    def remove_checkpoint(self, s3_uri: _PATH) -> None:
        """Remove checkpoint file from the path."""
        bucket, key = parse_s3_uri(s3_uri)
        self._client.delete_object(bucket, key)

    def teardown(self) -> None:
        """This method is called to teardown the process."""
        pass
