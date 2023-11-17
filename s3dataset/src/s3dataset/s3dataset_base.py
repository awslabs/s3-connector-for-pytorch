#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from functools import partial
from typing import (
    Iterable,
    Union,
    Tuple,
    Callable,
    Any,
    List,
)

from s3dataset._s3_bucket_iterable import S3BucketIterable
from s3dataset._s3client import S3Client, S3Reader

"""
s3dataset_base.py
    Base class for S3 datasets, containing logic for URIs parsing and objects listing. 
"""


def _identity(obj: S3Reader) -> S3Reader:
    return obj


class S3DatasetBase:
    def __init__(
        self,
        region: str,
        get_dataset_objects: Callable[[S3Client], Iterable[S3Reader]],
        transform: Callable[[S3Reader], Any] = _identity,
    ):
        self._get_dataset_objects = get_dataset_objects
        self._transform = transform
        self._region = region
        self._client = None

    @property
    def region(self):
        return self._region

    def _get_client(self):
        if self._client is None:
            self._client = S3Client(self.region)
        return self._client

    @classmethod
    def from_objects(
        cls,
        object_uris: Union[str, Iterable[str]],
        *,
        region: str,
        transform: Callable[[S3Reader], Any] = _identity,
    ):
        """
        Returns an instance of this dataset using the URI(s) provided.
        Args:
          object_uris(str or Iterable[str]):
            S3 URI of the object(s) desired.
          region(str or None):
            The S3 region where the objects are stored.
          transform:
            Optional callable which is used to transform an S3Reader into the desired type.
        """
        return cls(
            region, partial(_get_objects_from_uris, object_uris), transform=transform
        )

    @classmethod
    def from_prefix(
        cls,
        s3_uri: str,
        *,
        region: str,
        transform: Callable[[S3Reader], Any] = _identity,
    ):
        """
        Returns an instance of this dataset using the objects under bucket/prefix.
        Args:
          s3_uri(str):
            The S3 prefix (in the form of an s3_uri) for the objects in scope.
          region(str):
            The S3 region where the bucket is.
          transform:
            Optional callable which is used to transform an S3Reader into the desired type.
        """
        return cls(
            region, partial(_list_objects_from_prefix, s3_uri), transform=transform
        )


# TODO: Check boto3 implementation for this
def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    # TODO: We should be able to support more through Mountpoint, not sure if we want to
    if not uri or not uri.startswith("s3://"):
        raise ValueError("Only s3:// URIs are supported")
    uri = uri[len("s3://") :]
    if not uri:
        raise ValueError("Bucket name must be non-empty")
    split = uri.split("/", maxsplit=1)
    if len(split) == 1:
        bucket = split[0]
        prefix = ""
    else:
        bucket, prefix = split
    if not bucket:
        raise ValueError("Bucket name must be non-empty")
    return bucket, prefix


def _get_objects_from_uris(
    object_uris: Union[str, Iterable[str]], client: S3Client
) -> Iterable[S3Reader]:
    if isinstance(object_uris, str):
        object_uris = [object_uris]
    # TODO: We should be consistent with URIs parsing. Revise if we want to do this upfront or lazily.
    bucket_key_pairs = [_parse_s3_uri(uri) for uri in object_uris]

    return _bucket_key_pairs_to_objects(bucket_key_pairs, client)


def _bucket_key_pairs_to_objects(
    bucket_key_pairs: List[Tuple[str, str]], client: S3Client
):
    for bucket, key in bucket_key_pairs:
        yield client.get_object(bucket, key)


def _list_objects_from_prefix(s3_uri: str, client: S3Client) -> S3BucketIterable:
    bucket, prefix = _parse_s3_uri(s3_uri)
    return S3BucketIterable(client, bucket, prefix)
