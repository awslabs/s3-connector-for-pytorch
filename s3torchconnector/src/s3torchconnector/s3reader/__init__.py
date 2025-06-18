#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from .s3reader import S3Reader
from .s3reader_config import S3ReaderConfig
from .base import _BaseS3Reader
from .sequential import _SequentialS3Reader
from .ranged import _RangedS3Reader

__all__ = [
    'S3Reader',
    'S3ReaderConfig',
]
