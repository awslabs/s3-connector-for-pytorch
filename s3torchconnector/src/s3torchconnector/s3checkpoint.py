#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from s3torchconnector._s3dataset_common import parse_s3_uri
from s3torchconnector._s3client import S3Client, S3Reader, S3Writer


class S3Checkpoint:
    def __init__(self, region: str):
        self.region = region
        self._client = S3Client(region)

    def reader(self, s3_uri: str) -> S3Reader:
        bucket, key = parse_s3_uri(s3_uri)
        return self._client.get_object(bucket, key)

    def writer(self, s3_uri: str) -> S3Writer:
        bucket, key = parse_s3_uri(s3_uri)
        return self._client.put_object(bucket, key)
