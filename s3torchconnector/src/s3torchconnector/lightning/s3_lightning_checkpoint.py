#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from typing import Optional, Dict, Any

import lightning
import torch

from lightning.pytorch.plugins.io import CheckpointIO

from .._s3client import S3Client, S3ClientConfig
from .._s3dataset_common import parse_s3_uri
from .._user_agent import UserAgent


class S3LightningCheckpoint(CheckpointIO):
    """A checkpoint manager for S3 using the :class:`CheckpointIO` interface."""

    def __init__(
        self,
        region: str,
        s3client_config: Optional[S3ClientConfig] = None,
        endpoint: Optional[str] = None,
    ):
        self.region = region
        user_agent = UserAgent(["lightning", lightning.__version__])
        self._client = S3Client(
            region,
            user_agent=user_agent,
            s3client_config=s3client_config,
            endpoint=endpoint,
        )

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        # We only support `str` arguments for `path`, as `Path` is explicitly for local filesystems
        path: str,  # type: ignore
        storage_options: Optional[Any] = None,
    ) -> None:
        """Save model/training states as a checkpoint file through state-dump and upload to S3.

        Args:
            checkpoint (Dict[str, Any]): Containing model and trainer state
            path (str): Write-target S3 uri
            storage_options: Optional parameters when saving the model/training states.
        """
        self._validate_path(path)
        bucket, key = parse_s3_uri(path)
        with self._client.put_object(bucket, key) as s3writer:
            torch.save(checkpoint, s3writer)

    def load_checkpoint(
        self,
        # We only support `str` arguments for `path`, as `Path` is explicitly for local filesystems
        path: str,  # type: ignore
        map_location: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Load checkpoint from an S3 location when resuming or loading ckpt for test/validate/predict stages.

        Args:
            path (str): S3 uri to checkpoint
            map_location: A function, :class:`torch.device`, string or a dict specifying how to remap storage locations.

        Returns:
            Dict[str, Any]: The loaded checkpoint

        Raises:
            S3Exception: An error occurred accessing S3.
        """
        self._validate_path(path)
        bucket, key = parse_s3_uri(path)
        s3reader = self._client.get_object(bucket, key)
        # FIXME - io.BufferedIOBase and typing.IO aren't compatible
        #  See https://github.com/python/typeshed/issues/6077

        # Explicitly set weights_only=False to:
        # 1. Maintain backwards compatibility with older PyTorch versions where this was the default behavior
        # 2. Match PyTorch Lightning's implementation strategy for consistent behavior
        # Reference: https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/fabric/utilities/cloud_io.py#L36
        return torch.load(s3reader, map_location, weights_only=False)  # type: ignore

    def remove_checkpoint(
        self,
        # We only support `str` arguments for `path`, as `Path` is explicitly for local filesystems
        path: str,  # type: ignore
    ) -> None:
        """Remove checkpoint file from the S3 uri.

        Args:
            path (str): S3 uri to checkpoint

        Raises:
            S3Exception: An error occurred accessing S3.
        """
        self._validate_path(path)
        bucket, key = parse_s3_uri(path)
        self._client.delete_object(bucket, key)

    def teardown(self) -> None:
        """This method is called to teardown the process."""
        pass

    @staticmethod
    def _validate_path(path: str) -> None:
        if not isinstance(path, str):
            raise TypeError(
                f"{type(path).__name__!r} is not a supported type for 'path'. Must be a string formatted as an S3 uri."
            )
