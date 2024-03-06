#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
from __future__ import annotations

from functools import partial
from typing import (
    Iterator,
    Union,
    Iterable,
    Callable,
    Optional,
    TypeVar,
    cast,
    overload,
)
import logging

import torch.utils.data

from . import S3Reader
from ._s3bucket_key_data import S3BucketKeyData
from ._s3client import S3Client
from ._s3dataset_common import (
    identity,
    get_objects_from_uris,
    get_objects_from_prefix,
)

log = logging.getLogger(__name__)

T_co = TypeVar("T_co", covariant=True)


class S3IterableDataset(torch.utils.data.IterableDataset[T_co]):
    """An IterableStyle dataset created from S3 objects.

    To create an instance of S3IterableDataset, you need to use
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

    @property
    def region(self) -> str:
        return self._region

    @property
    def endpoint(self) -> Optional[str]:
        return self._endpoint

    @overload
    @classmethod
    def from_objects(
        cls,
        object_uris: Union[str, Iterable[str]],
        *,
        region: str,
        endpoint: Optional[str] = None,
    ) -> S3IterableDataset[S3Reader]: ...

    @overload
    @classmethod
    def from_objects(
        cls,
        object_uris: Union[str, Iterable[str]],
        *,
        region: str,
        endpoint: Optional[str] = None,
        transform: Callable[[S3Reader], T_co],
    ) -> S3IterableDataset[T_co]: ...

    @classmethod
    def from_objects(
        cls,
        object_uris: Union[str, Iterable[str]],
        *,
        region: str,
        endpoint: Optional[str] = None,
        transform: Callable[[S3Reader], T_co | S3Reader] = identity,
    ) -> S3IterableDataset[T_co | S3Reader]:
        """Returns an instance of S3IterableDataset using the S3 URI(s) provided.

        Args:
          object_uris(str | Iterable[str]): S3 URI of the object(s) desired.
          region(str): AWS region of the S3 bucket where the objects are stored.
          endpoint(str): AWS endpoint of the S3 bucket where the objects are stored.
          transform: Optional callable which is used to transform an S3Reader into the desired type.

        Returns:
            S3IterableDataset: An IterableStyle dataset created from S3 objects.

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
    ) -> S3IterableDataset[S3Reader]: ...

    @overload
    @classmethod
    def from_prefix(
        cls,
        s3_uri: str,
        *,
        region: str,
        endpoint: Optional[str] = None,
        transform: Callable[[S3Reader], T_co],
    ) -> S3IterableDataset[T_co]: ...

    @classmethod
    def from_prefix(
        cls,
        s3_uri: str,
        *,
        region: str,
        endpoint: Optional[str] = None,
        transform: Callable[[S3Reader], T_co | S3Reader] = identity,
    ) -> S3IterableDataset[T_co | S3Reader]:
        """Returns an instance of S3IterableDataset using the S3 URI provided.

        Args:
          s3_uri(str): An S3 URI (prefix) of the object(s) desired. Objects matching the prefix will be included in the returned dataset.
          region(str): AWS region of the S3 bucket where the objects are stored.
          endpoint(str): AWS endpoint of the S3 bucket where the objects are stored.
          transform: Optional callable which is used to transform an S3Reader into the desired type.

        Returns:
            S3IterableDataset: An IterableStyle dataset created from S3 objects.

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

    def _get_transformed_object(self, bucket_key: S3BucketKeyData) -> T_co:
        return self._transform(
            self._get_client().get_object(
                bucket_key.bucket, bucket_key.key, object_info=bucket_key.object_info
            )
        )

    def __iter__(self) -> Iterator[T_co]:
        return map(
            self._get_transformed_object, self._get_dataset_objects(self._get_client())
        )
