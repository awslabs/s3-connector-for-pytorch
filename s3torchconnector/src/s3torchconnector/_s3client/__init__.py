#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from ._s3client import S3Client
from ._mock_s3client import MockS3Client

__all__ = ["S3Client", "MockS3Client"]
