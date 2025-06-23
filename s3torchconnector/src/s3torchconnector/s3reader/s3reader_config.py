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

    Usage Examples:
        # For sequential reading (default)
        config = S3ReaderConfig()  # or S3ReaderConfig.sequential()

        # For range-based reading
        config = S3ReaderConfig.range_based()
    """

    class ReaderType(Enum):
        SEQUENTIAL = "sequential"
        RANGE_BASED = "range_based"

    # Default to SEQUENTIAL reader
    reader_type: ReaderType = ReaderType.SEQUENTIAL

    @classmethod
    def sequential(cls) -> S3ReaderConfig:
        """Alternative constructor for sequential reading configuration."""
        return cls(reader_type=cls.ReaderType.SEQUENTIAL)

    @classmethod
    def range_based(cls) -> S3ReaderConfig:
        """Alternative constructor for range-based reading configuration."""
        return cls(reader_type=cls.ReaderType.RANGE_BASED)
    
    def get_reader_type_string(self) -> str:
        """Returns the lowercase string representation of the reader type."""
        return self.reader_type.value
