#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
from s3torchconnectorclient import S3Exception

# The order of these imports is the same in which they will be rendered
# in the API docs generated with Sphinx.

from .s3reader import S3Reader, S3ReaderConstructor
from .s3writer import S3Writer
from .s3iterable_dataset import S3IterableDataset
from .s3map_dataset import S3MapDataset
from .s3checkpoint import S3Checkpoint
from ._version import __version__
from ._s3client import S3ClientConfig

__all__ = [
    "S3IterableDataset",
    "S3MapDataset",
    "S3Checkpoint",
    "S3Reader",
    "S3ReaderConstructor",
    "S3Writer",
    "S3Exception",
    "S3ClientConfig",
    "__version__",
]
