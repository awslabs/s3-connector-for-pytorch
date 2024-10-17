#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
from functools import partial
from typing import Iterator, Any, Union, Iterable, Callable, Optional
import logging

import torch.utils.data
import torch

from . import S3Reader
from ._s3bucket_key_data import S3BucketKeyData
from ._s3client import S3Client, S3ClientConfig
from ._s3dataset_common import (
    identity,
    get_objects_from_uris,
    get_objects_from_prefix,
)

log = logging.getLogger(__name__)


class S3IterableDataset(torch.utils.data.IterableDataset):
    """An IterableStyle dataset created from S3 objects.

    To create an instance of S3IterableDataset, you need to use
    `from_prefix` or `from_objects` methods.
    """

    def __init__(
        self,
        region: str,
        get_dataset_objects: Callable[[S3Client], Iterable[S3BucketKeyData]],
        endpoint: Optional[str] = None,
        transform: Callable[[S3Reader], Any] = identity,
        s3client_config: Optional[S3ClientConfig] = None,
        share_dataset_within_process: bool = False,
    ):
        self._get_dataset_objects = get_dataset_objects
        self._transform = transform
        self._region = region
        self._endpoint = endpoint
        self._s3client_config = s3client_config
        self._client = None
        self._share_dataset_within_process = share_dataset_within_process

        self._rank = 0
        self._world_size = 1
        if torch.distributed.is_initialized():
            self._rank = torch.distributed.get_rank()
            self._world_size = torch.distributed.get_world_size()

    @property
    def region(self):
        return self._region

    @property
    def endpoint(self):
        return self._endpoint

    @classmethod
    def from_objects(
        cls,
        object_uris: Union[str, Iterable[str]],
        *,
        region: str,
        endpoint: Optional[str] = None,
        transform: Callable[[S3Reader], Any] = identity,
        s3client_config: Optional[S3ClientConfig] = None,
        share_dataset_within_process: bool = False,
    ):
        """Returns an instance of S3IterableDataset using the S3 URI(s) provided.

        Args:
          object_uris(str | Iterable[str]): S3 URI of the object(s) desired.
          region(str): AWS region of the S3 bucket where the objects are stored.
          endpoint(str): AWS endpoint of the S3 bucket where the objects are stored.
          transform: Optional callable which is used to transform an S3Reader into the desired type.
          s3client_config: Optional S3ClientConfig with parameters for S3 client.
          share_dataset_within_process: share the dataset across workers within the same process, but use different datasets for different processes. Turned off by default.

        Returns:
            S3IterableDataset: An IterableStyle dataset created from S3 objects.

        Raises:
            S3Exception: An error occurred accessing S3.
        """
        log.info(f"Building {cls.__name__} from_objects")
        return cls(
            region,
            partial(get_objects_from_uris, object_uris),
            endpoint,
            transform=transform,
            s3client_config=s3client_config,
            share_dataset_within_process=share_dataset_within_process,
        )

    @classmethod
    def from_prefix(
        cls,
        s3_uri: str,
        *,
        region: str,
        endpoint: Optional[str] = None,
        transform: Callable[[S3Reader], Any] = identity,
        s3client_config: Optional[S3ClientConfig] = None,
        share_dataset_within_process: bool = False,
    ):
        """Returns an instance of S3IterableDataset using the S3 URI provided.

        Args:
          s3_uri(str): An S3 URI (prefix) of the object(s) desired. Objects matching the prefix will be included in the returned dataset.
          region(str): AWS region of the S3 bucket where the objects are stored.
          endpoint(str): AWS endpoint of the S3 bucket where the objects are stored.
          transform: Optional callable which is used to transform an S3Reader into the desired type.
          s3client_config: Optional S3ClientConfig with parameters for S3 client.
          share_dataset_within_process: share the dataset across workers within the same process, but use different datasets for different processes. Turned off by default.

        Returns:
            S3IterableDataset: An IterableStyle dataset created from S3 objects.

        Raises:
            S3Exception: An error occurred accessing S3.
        """
        log.info(f"Building {cls.__name__} from_prefix {s3_uri=}")
        return cls(
            region,
            partial(get_objects_from_prefix, s3_uri),
            endpoint,
            transform=transform,
            s3client_config=s3client_config,
            share_dataset_within_process=share_dataset_within_process,
        )

    def _get_client(self):
        if self._client is None:
            self._client = S3Client(
                self.region,
                endpoint=self.endpoint,
                s3client_config=self._s3client_config,
            )
        return self._client

    def _get_transformed_object(self, bucket_key: S3BucketKeyData) -> Any:
        return self._transform(
            self._get_client().get_object(
                bucket_key.bucket, bucket_key.key, object_info=bucket_key.object_info
            )
        )

    def __iter__(self) -> Iterator[Any]:
        worker_id = 0
        num_workers = 1
        if not self._share_dataset_within_process:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                worker_id = worker_info.id
                num_workers = worker_info.num_workers

        self._shard_index = num_workers * self._rank + worker_id
        self._shard_count = num_workers * self._world_size

        if self._shard_index == 0 and self._shard_count == 1:
            return map(
                self._get_transformed_object,
                self._get_dataset_objects(self._get_client()),
            )

        sharded_objects = (
            obj
            for idx, obj in enumerate(self._get_dataset_objects(self._get_client()))
            if idx % self._shard_count == self._shard_index
        )
        return map(self._get_transformed_object, sharded_objects)
