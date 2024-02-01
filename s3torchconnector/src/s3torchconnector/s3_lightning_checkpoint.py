import torch
from lightning.pytorch.plugins.io import CheckpointIO

from typing import *

from s3torchconnector._s3client import S3Client
from lightning.fabric.utilities.types import _PATH

from s3torchconnector._s3dataset_common import parse_s3_uri

class S3LightningCheckpoint(CheckpointIO):
    def __init__(self, region: str):
        self.region = region
        self._client = S3Client(region)

    def save_checkpoint(self, checkpoint: Dict[str, Any], s3_uri: _PATH, storage_options: Optional[Any] = None) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write."""
        bucket, key = parse_s3_uri(s3_uri)
        s3writer = self._client.put_object(bucket, key)
        torch.save(checkpoint, s3writer)
        s3writer.close()

    def load_checkpoint(self, s3_uri: _PATH, map_location: Optional[Any] = None) -> Dict[str, Any]:
        """Load checkpoint from a path when resuming or loading ckpt for test/validate/predict stages."""
        bucket, key = parse_s3_uri(s3_uri)
        s3reader = self._client.get_object(bucket, key)
        return torch.load(s3reader, map_location)

    def remove_checkpoint(self, s3_uri: _PATH) -> None:
        """Remove checkpoint file from the path."""
        pass

    def teardown(self) -> None:
        """This method is called to teardown the process."""
        pass