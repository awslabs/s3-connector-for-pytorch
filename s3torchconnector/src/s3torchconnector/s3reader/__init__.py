#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from .s3reader import S3Reader
from .constructor import S3ReaderConstructor, DCPListOfRangesConstructor
from .sequential import SequentialS3Reader
from .ranged import RangedS3Reader
from .list_of_ranges import ListOfRangesS3Reader
from .protocol import GetStreamCallable, S3ReaderConstructorProtocol

__all__ = [
    "S3Reader",
    "S3ReaderConstructor",
    "SequentialS3Reader",
    "RangedS3Reader",
    "ListOfRangesS3Reader",
]
