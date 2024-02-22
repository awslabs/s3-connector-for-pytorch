#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from functools import partial
from itertools import chain
from typing import Iterator, List

from s3torchconnectorclient._mountpoint_s3_client import (
    ObjectInfo,
    ListObjectResult,
    ListObjectStream,
)

from ._s3bucket_key_data import S3BucketKeyData
from ._s3client import S3Client


class S3BucketIterable:
    def __init__(self, client: S3Client, bucket: str, prefix: str):
        self._client = client
        self._bucket = bucket
        self._prefix = prefix

    def __iter__(self) -> Iterator[S3BucketKeyData]:
        # This allows us to iterate multiple times by re-creating the `_list_stream`
        return iter(S3BucketIterator(self._client, self._bucket, self._prefix))


class S3BucketIterator:
    def __init__(self, client: S3Client, bucket: str, prefix: str):
        self._client = client
        self._bucket = bucket
        self._list_stream = _PickleableListObjectStream(client, bucket, prefix)

    def __iter__(self) -> Iterator[S3BucketKeyData]:
        return chain.from_iterable(
            map(partial(_extract_list_results, self._bucket), self._list_stream)
        )


class _PickleableListObjectStream:
    def __init__(self, client: S3Client, bucket: str, prefix: str):
        self._client = client
        self._list_stream = iter(client.list_objects(bucket, prefix))

    def __iter__(self):
        return self

    def __next__(self) -> ListObjectResult:
        return next(self._list_stream)

    def __getstate__(self):
        return {
            "client": self._client,
            "bucket": self._list_stream.bucket,
            "prefix": self._list_stream.prefix,
            "delimiter": self._list_stream.delimiter,
            "max_keys": self._list_stream.max_keys,
            "continuation_token": self._list_stream.continuation_token,
            "complete": self._list_stream.complete,
        }

    def __setstate__(self, state):
        self._client = state["client"]
        self._list_stream = ListObjectStream._from_state(**state)


def _extract_list_results(
    bucket: str, list_result: ListObjectResult
) -> Iterator[S3BucketKeyData]:
    return map(partial(_extract_object_info, bucket), list_result.object_info)


def _extract_object_info(bucket: str, object_info: ObjectInfo) -> S3BucketKeyData:
    return S3BucketKeyData(bucket=bucket, key=object_info.key, object_info=object_info)
