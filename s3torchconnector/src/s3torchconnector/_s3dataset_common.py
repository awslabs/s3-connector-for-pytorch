#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from typing import (
    Iterable,
    Union,
    Tuple,
    List,
)

from s3torchconnector._s3_bucket_iterable import S3BucketIterable
from s3torchconnector._s3client import S3Client, S3Reader

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
) -> Iterable[S3Reader]:
    if isinstance(object_uris, str):
        object_uris = [object_uris]
    # TODO: We should be consistent with URIs parsing. Revise if we want to do this upfront or lazily.
    bucket_key_pairs = [parse_s3_uri(uri) for uri in object_uris]

    return bucket_key_pairs_to_objects(bucket_key_pairs, client)


def bucket_key_pairs_to_objects(
    bucket_key_pairs: List[Tuple[str, str]], client: S3Client
):
    for bucket, key in bucket_key_pairs:
        yield client.get_object(bucket, key)


def list_objects_from_prefix(s3_uri: str, client: S3Client) -> S3BucketIterable:
    bucket, prefix = parse_s3_uri(s3_uri)
    return S3BucketIterable(client, bucket, prefix)
