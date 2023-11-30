#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from ._s3client import S3Client
from .s3reader import S3Reader
from .s3writer import S3Writer
from ._mock_s3client import MockS3Client

__all__ = ["S3Client", "MockS3Client", "S3Reader", "S3Writer"]
