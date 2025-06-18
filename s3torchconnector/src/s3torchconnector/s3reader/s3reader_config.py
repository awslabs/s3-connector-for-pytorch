#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

@dataclass(frozen=True)
class S3ReaderConfig:
    """A dataclass exposing configurable parameters for S3Reader.

    Args:
    reader_type (ReaderType): Determines the S3 access strategy.
        SEQUENTIAL:  Buffers entire object sequentially. Best for full reads and repeated access.
        RANGE_BASED: Fetches specific byte ranges on-demand. Suitable for partial reads of large objects.
    """

    class ReaderType(Enum):
        SEQUENTIAL = "sequential"
        RANGE_BASED = "range"

    # Default to SequentialS3Reader
    reader_type: ReaderType = ReaderType.SEQUENTIAL

    @classmethod
    def sequential(cls) -> S3ReaderConfig:
        """Alternative constructor for sequential reading configuration."""
        return cls(reader_type=cls.ReaderType.SEQUENTIAL)

    @classmethod
    def range_based(cls) -> S3ReaderConfig:
        """Alternative constructor for range-based reading configuration."""
        return cls(reader_type=cls.ReaderType.RANGE_BASED)