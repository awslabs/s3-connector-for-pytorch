#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
from functools import partial
from typing import List, Any, Callable, Iterable, Union, Optional
import logging

import torch.utils.data
from s3torchconnector._s3bucket_key_data import S3BucketKeyData

from ._s3client import S3Client, S3ClientConfig
from . import S3Reader, S3ReaderConstructor
from .s3reader import S3ReaderConstructorProtocol
from ._user_agent import UserAgent

from ._s3dataset_common import (
    get_objects_from_uris,
    get_objects_from_prefix,
    identity,
)

log = logging.getLogger(__name__)


class S3MapDataset(torch.utils.data.Dataset):
    """A Map-Style dataset created from S3 objects.

    To create an instance of S3MapDataset, you need to use
    `from_prefix` or `from_objects` methods.
    """

    def __init__(
        self,
        region: str,
        get_dataset_objects: Callable[[S3Client], Iterable[S3BucketKeyData]],
        endpoint: Optional[str] = None,
        transform: Callable[[S3Reader], Any] = identity,
        s3client_config: Optional[S3ClientConfig] = None,
        reader_constructor: Optional[S3ReaderConstructorProtocol] = None,
    ):
        self._get_dataset_objects = get_dataset_objects
        self._transform = transform
        self._region = region
        self._endpoint = endpoint
        self._s3client_config = s3client_config
        self._client = None
        self._bucket_key_pairs: Optional[List[S3BucketKeyData]] = None
        self._reader_constructor = reader_constructor or S3ReaderConstructor.default()

    @property
    def region(self):
        return self._region

    @property
    def endpoint(self):
        return self._endpoint

    @property
    def _dataset_bucket_key_pairs(self) -> List[S3BucketKeyData]:
        if self._bucket_key_pairs is None:
            self._bucket_key_pairs = list(self._get_dataset_objects(self._get_client()))
        assert self._bucket_key_pairs is not None
        return self._bucket_key_pairs

    @classmethod
    def from_objects(
        cls,
        object_uris: Union[str, Iterable[str]],
        *,
        region: str,
        endpoint: Optional[str] = None,
        transform: Callable[[S3Reader], Any] = identity,
        s3client_config: Optional[S3ClientConfig] = None,
        reader_constructor: Optional[S3ReaderConstructorProtocol] = None,
    ):
        """Returns an instance of S3MapDataset using the S3 URI(s) provided.

        Args:
          object_uris(str | Iterable[str]): S3 URI of the object(s) desired.
          region(str): AWS region of the S3 bucket where the objects are stored.
          endpoint(str): AWS endpoint of the S3 bucket where the objects are stored.
          transform: Optional callable which is used to transform an S3Reader into the desired type.
          s3client_config: Optional S3ClientConfig with parameters for S3 client.
          reader_constructor (Optional[S3ReaderConstructorProtocol]): Optional partial(S3Reader) created using S3ReaderConstructor
            e.g. S3ReaderConstructor.sequential() or S3ReaderConstructor.range_based()

        Returns:
            S3MapDataset: A Map-Style dataset created from S3 objects.

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
            reader_constructor=reader_constructor,
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
        reader_constructor: Optional[S3ReaderConstructorProtocol] = None,
    ):
        """Returns an instance of S3MapDataset using the S3 URI provided.

        Args:
          s3_uri(str): An S3 URI (prefix) of the object(s) desired. Objects matching the prefix will be included in the returned dataset.
          region(str): AWS region of the S3 bucket where the objects are stored.
          endpoint(str): AWS endpoint of the S3 bucket where the objects are stored.
          transform: Optional callable which is used to transform an S3Reader into the desired type.
          s3client_config: Optional S3ClientConfig with parameters for S3 client.
          reader_constructor (Optional[S3ReaderConstructorProtocol]): Optional partial(S3Reader) created using S3ReaderConstructor
            e.g. S3ReaderConstructor.sequential() or S3ReaderConstructor.range_based()

        Returns:
            S3MapDataset: A Map-Style dataset created from S3 objects.

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
            reader_constructor=reader_constructor,
        )

    def _get_client(self):
        if self._client is None:
            reader_type_string = S3ReaderConstructor.get_reader_type_string(
                self._reader_constructor
            )
            self._client = S3Client(
                self.region,
                endpoint=self.endpoint,
                user_agent=UserAgent(
                    comments=[f"md/dataset#map md/reader_type#{reader_type_string}"]
                ),
                s3client_config=self._s3client_config,
            )
        return self._client

    def _get_object(self, i: int) -> S3Reader:
        bucket_key = self._dataset_bucket_key_pairs[i]
        return self._get_client().get_object(
            bucket_key.bucket,
            bucket_key.key,
            object_info=bucket_key.object_info,
            reader_constructor=self._reader_constructor,
        )

    def __getitem__(self, i: int) -> Any:
        return self._transform(self._get_object(i))

    def __len__(self):
        return len(self._dataset_bucket_key_pairs)
