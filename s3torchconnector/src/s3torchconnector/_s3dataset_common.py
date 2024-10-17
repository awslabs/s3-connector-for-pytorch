#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from typing import Iterable, Union, Tuple

from ._s3_bucket_iterable import S3BucketIterable
from ._s3client import S3Client
from . import S3Reader
from ._s3bucket_key_data import S3BucketKeyData

"""
_s3dataset_common.py
    Collection of common methods for S3 datasets, containing logic for URIs parsing and objects listing. 
"""


def identity(obj: S3Reader) -> S3Reader:
    return obj


# TODO: Check boto3 implementation for this
def parse_s3_uri(uri: str) -> Tuple[str, str]:
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


def get_objects_from_uris(
    object_uris: Union[str, Iterable[str]], client: S3Client
) -> Iterable[S3BucketKeyData]:
    if isinstance(object_uris, str):
        object_uris = [object_uris]
    # TODO: We should be consistent with URIs parsing. Revise if we want to do this upfront or lazily.
    bucket_key_pairs = [parse_s3_uri(uri) for uri in object_uris]

    return (S3BucketKeyData(bucket, key) for bucket, key in bucket_key_pairs)


def get_objects_from_prefix(s3_uri: str, client: S3Client) -> Iterable[S3BucketKeyData]:
    bucket, prefix = parse_s3_uri(s3_uri)
    return iter(S3BucketIterable(client, bucket, prefix))
