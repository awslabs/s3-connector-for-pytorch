#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
from functools import partial
from typing import Iterator, Any, Union, Iterable, Callable

import torch.utils.data

from . import S3Reader
from ._s3client import S3Client
from ._s3dataset_common import (
    identity,
    get_objects_from_uris,
    list_objects_from_prefix,
)

"""
s3iterable_dataset.py
    API for accessing as PyTorch IterableDataset files stored in S3. 
"""


class S3IterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        region: str,
        get_dataset_objects: Callable[[S3Client], Iterable[S3Reader]],
        transform: Callable[[S3Reader], Any] = identity,
    ):
        self._get_dataset_objects = get_dataset_objects
        self._transform = transform
        self._region = region
        self._client = None

    @property
    def region(self):
        return self._region

    @classmethod
    def from_objects(
        cls,
        object_uris: Union[str, Iterable[str]],
        *,
        region: str,
        transform: Callable[[S3Reader], Any] = identity,
    ):
        """
        Returns an instance of S3IterableDataset dataset using the URI(s) provided.
        Args:
          object_uris(str or Iterable[str]):
            S3 URI of the object(s) desired.
          region(str):
            The S3 region where the objects are stored.
          transform:
            Optional callable which is used to transform an S3Reader into the desired type.
        """
        return cls(
            region, partial(get_objects_from_uris, object_uris), transform=transform
        )

    @classmethod
    def from_prefix(
        cls,
        s3_uri: str,
        *,
        region: str,
        transform: Callable[[S3Reader], Any] = identity,
    ):
        """
        Returns an instance of S3IterableDataset dataset using the objects under bucket/prefix.
        Args:
          s3_uri(str):
            The S3 prefix (in the form of an s3_uri) for the objects in scope.
          region(str):
            The S3 region where the bucket is.
          transform:
            Optional callable which is used to transform an S3Reader into the desired type.
        """
        return cls(
            region, partial(list_objects_from_prefix, s3_uri), transform=transform
        )

    def _get_client(self):
        if self._client is None:
            self._client = S3Client(self.region)
        return self._client

    def __iter__(self) -> Iterator[Any]:
        return map(self._transform, self._get_dataset_objects(self._get_client()))
