#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
from functools import partial
from typing import List, Any, Callable, Iterable, Union

import torch.utils.data

from ._s3client import S3Client, S3Reader

from ._s3dataset_common import (
    get_objects_from_uris,
    list_objects_from_prefix,
    identity,
)


class S3MapDataset(torch.utils.data.Dataset):
    """A Map-Style dataset created from S3 objects.

    To create an instance of S3MapDataset, you need to use
    `from_prefix` or `from_objects` methods.
    """

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
        self._dataset_object_store = None

    @property
    def region(self):
        return self._region

    @property
    def _dataset_objects(self) -> List[S3Reader]:
        if self._dataset_object_store is None:
            self._dataset_object_store = list(
                self._get_dataset_objects(self._get_client())
            )
        return self._dataset_object_store

    @classmethod
    def from_objects(
        cls,
        object_uris: Union[str, Iterable[str]],
        *,
        region: str,
        transform: Callable[[S3Reader], Any] = identity,
    ):
        """Returns an instance of S3MapDataset using the S3 URI(s) provided.

        Args:
          object_uris(str | Iterable[str]): S3 URI of the object(s) desired.
          region(str): AWS region of the S3 bucket where the objects are stored.
          transform: Optional callable which is used to transform an S3Reader into the desired type.

        Returns:
            S3MapDataset: A Map-Style dataset created from S3 objects.

        Raises:
            S3Exception: An error occurred accessing S3.
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
        """Returns an instance of S3MapDataset using the S3 URI provided.

        Args:
          s3_uri(str): An S3 URI (prefix) of the object(s) desired. Objects matching the prefix will be included in the returned dataset.
          region(str): AWS region of the S3 bucket where the objects are stored.
          transform: Optional callable which is used to transform an S3Reader into the desired type.

        Returns:
            S3MapDataset: A Map-Style dataset created from S3 objects.

        Raises:
            S3Exception: An error occurred accessing S3.
        """
        return cls(
            region, partial(list_objects_from_prefix, s3_uri), transform=transform
        )

    def _get_client(self):
        if self._client is None:
            self._client = S3Client(self.region)
        return self._client

    def __getitem__(self, i: int) -> Any:
        return self._transform(self._dataset_objects[i])

    def __len__(self):
        return len(self._dataset_objects)
