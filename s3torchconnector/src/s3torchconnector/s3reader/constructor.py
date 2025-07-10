#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from functools import partial
from typing import Optional

from .protocol import S3ReaderConstructorProtocol
from .sequential import SequentialS3Reader
from .ranged import RangedS3Reader


class S3ReaderConstructor:
    """Constructor for creating ``partial(S3Reader)`` instances.

    Creates partial ``S3Reader`` instances that will be completed by ``S3Client`` with the
    remaining required parameters (e.g. ``bucket``, ``key``, ``get_object_info``, ``get_stream``).

    The constructor provides factory methods for different reader types:

    - ``sequential()``: Creates a constructor for sequential readers that buffer the entire object.
      Best for full reads and repeated access.
    - ``range_based()``: Creates a constructor for range-based readers that fetch specific byte ranges.
      Suitable for sparse partial reads for large objects.
    """

    @staticmethod
    def sequential() -> S3ReaderConstructorProtocol:
        """Creates a constructor for sequential readers

        Returns:
            S3ReaderConstructorProtocol: Partial constructor for SequentialS3Reader

        Example::

            reader_constructor = S3ReaderConstructor.sequential()

        """
        return partial(SequentialS3Reader)

    @staticmethod
    def range_based(buffer_size: Optional[int] = None) -> S3ReaderConstructorProtocol:
        """Creates a constructor for range-based readers

        Args:
            buffer_size: Internal buffer size in bytes. If None, uses default 8MB.
                         Set to 0 to disable buffering.

        Returns:
            S3ReaderConstructorProtocol: Partial constructor for RangedS3Reader

        Range-based reader performs byte-range requests to read specific portions of S3 objects without
        downloading the entire file.

        Buffer size affects read performance:

        * Small reads (< ``buffer_size``): Loads ``buffer_size`` bytes to buffer to reduce S3 API calls for small, sequential reads
        * Large reads (≥ ``buffer_size``): bypass the buffer for direct transfer from S3
        * Forward overlap reads: Reuses buffered data when reading ranges that extend beyond current buffer, and processes remaining
        data according to size with logic above.

        Configuration Guide:

        * Use larger buffer sizes for workloads with many small, sequential reads of nearby bytes
        * Use smaller buffer sizes or disable buffering for sparse partial reads
        * Buffer can be disabled by setting ``buffer_size`` to 0
        * If ``buffer_size`` is None, uses default 8MB buffer

        Examples::

            # Range-based reader with default 8MB buffer
            reader_constructor = S3ReaderConstructor.range_based()

            # Range-based reader with custom buffer size
            reader_constructor = S3ReaderConstructor.range_based(buffer_size=16*1024*1024)

            # Range-based reader with buffering disabled
            reader_constructor = S3ReaderConstructor.range_based(buffer_size=0)
        """
        return partial(RangedS3Reader, buffer_size=buffer_size)

    @staticmethod
    def default() -> S3ReaderConstructorProtocol:
        """Creates default reader constructor (sequential)

        Returns:
            S3ReaderConstructorProtocol: Partial constructor for SequentialS3Reader
        """
        return S3ReaderConstructor.sequential()

    @staticmethod
    def get_reader_type_string(
        constructor: Optional[S3ReaderConstructorProtocol],
    ) -> str:
        """Returns the reader type string for the given constructor."""
        if constructor is None:
            return S3ReaderConstructor.get_reader_type_string(
                S3ReaderConstructor.default()
            )

        if not isinstance(constructor, partial):
            return "unknown"

        if constructor.func == RangedS3Reader:
            return "range_based"
        elif constructor.func == SequentialS3Reader:
            return "sequential"
        else:
            return "unknown"
