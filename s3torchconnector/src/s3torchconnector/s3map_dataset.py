#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
from __future__ import annotations

from functools import partial
from typing import List, Callable, Iterable, Union, Optional, TypeVar, overload, cast
import logging

import torch.utils.data
from s3torchconnector._s3bucket_key_data import S3BucketKeyData

from ._s3client import S3Client
from . import S3Reader

from ._s3dataset_common import (
    get_objects_from_uris,
    get_objects_from_prefix,
    identity,
)

log = logging.getLogger(__name__)


T_co = TypeVar("T_co", covariant=True)


class S3MapDataset(torch.utils.data.Dataset[T_co]):
    """A Map-Style dataset created from S3 objects.

    To create an instance of S3MapDataset, you need to use
    `from_prefix` or `from_objects` methods.
    """

    def __init__(
        self,
        region: str,
        *,
        get_dataset_objects: Callable[[S3Client], Iterable[S3BucketKeyData]],
        endpoint: Optional[str] = None,
        transform: Callable[[S3Reader], T_co],
    ):
        self._get_dataset_objects = get_dataset_objects
        self._transform = transform
        self._region = region
        self._endpoint = endpoint
        self._client: Optional[S3Client] = None
        self._bucket_key_pairs: Optional[List[S3BucketKeyData]] = None

    @property
    def region(self) -> str:
        return self._region

    @property
    def endpoint(self) -> Optional[str]:
        return self._endpoint

    @property
    def _dataset_bucket_key_pairs(self) -> List[S3BucketKeyData]:
        if self._bucket_key_pairs is None:
            self._bucket_key_pairs = list(self._get_dataset_objects(self._get_client()))
        assert self._bucket_key_pairs is not None
        return self._bucket_key_pairs

    @overload
    @classmethod
    def from_objects(
        cls,
        object_uris: Union[str, Iterable[str]],
        *,
        region: str,
        endpoint: Optional[str] = None,
    ) -> S3MapDataset[S3Reader]: ...

    @overload
    @classmethod
    def from_objects(
        cls,
        object_uris: Union[str, Iterable[str]],
        *,
        region: str,
        endpoint: Optional[str] = None,
        transform: Callable[[S3Reader], T_co],
    ) -> S3MapDataset[T_co]: ...

    @classmethod
    def from_objects(
        cls,
        object_uris: Union[str, Iterable[str]],
        *,
        region: str,
        endpoint: Optional[str] = None,
        transform: Callable[[S3Reader], T_co | S3Reader] = identity,
    ) -> S3MapDataset[T_co | S3Reader]:
        """Returns an instance of S3MapDataset using the S3 URI(s) provided.

        Args:
          object_uris(str | Iterable[str]): S3 URI of the object(s) desired.
          region(str): AWS region of the S3 bucket where the objects are stored.
          endpoint(str): AWS endpoint of the S3 bucket where the objects are stored.
          transform: Optional callable which is used to transform an S3Reader into the desired type.

        Returns:
            S3MapDataset: A Map-Style dataset created from S3 objects.

        Raises:
            S3Exception: An error occurred accessing S3.
        """
        log.info(f"Building {cls.__name__} from_objects")
        return cls(
            region,
            get_dataset_objects=partial(get_objects_from_uris, object_uris),
            endpoint=endpoint,
            transform=cast(Callable[[S3Reader], T_co], transform),
        )

    @overload
    @classmethod
    def from_prefix(
        cls,
        s3_uri: str,
        *,
        region: str,
        endpoint: Optional[str] = None,
    ) -> S3MapDataset[S3Reader]: ...

    @overload
    @classmethod
    def from_prefix(
        cls,
        s3_uri: str,
        *,
        region: str,
        endpoint: Optional[str] = None,
        transform: Callable[[S3Reader], T_co],
    ) -> S3MapDataset[T_co]: ...

    @classmethod
    def from_prefix(
        cls,
        s3_uri: str,
        *,
        region: str,
        endpoint: Optional[str] = None,
        transform: Callable[[S3Reader], T_co | S3Reader] = identity,
    ) -> S3MapDataset[T_co | S3Reader]:
        """Returns an instance of S3MapDataset using the S3 URI provided.

        Args:
          s3_uri(str): An S3 URI (prefix) of the object(s) desired. Objects matching the prefix will be included in the returned dataset.
          region(str): AWS region of the S3 bucket where the objects are stored.
          endpoint(str): AWS endpoint of the S3 bucket where the objects are stored.
          transform: Optional callable which is used to transform an S3Reader into the desired type.

        Returns:
            S3MapDataset: A Map-Style dataset created from S3 objects.

        Raises:
            S3Exception: An error occurred accessing S3.
        """
        log.info(f"Building {cls.__name__} from_prefix {s3_uri=}")
        return cls(
            region,
            get_dataset_objects=partial(get_objects_from_prefix, s3_uri),
            endpoint=endpoint,
            transform=cast(Callable[[S3Reader], T_co], transform),
        )

    def _get_client(self) -> S3Client:
        if self._client is None:
            self._client = S3Client(self.region, self.endpoint)
        return self._client

    def _get_object(self, i: int) -> S3Reader:
        bucket_key = self._dataset_bucket_key_pairs[i]
        return self._get_client().get_object(
            bucket_key.bucket, bucket_key.key, object_info=bucket_key.object_info
        )

    def __getitem__(self, i: int) -> T_co:
        return self._transform(self._get_object(i))

    def __len__(self) -> int:
        return len(self._dataset_bucket_key_pairs)
