#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from functools import partial
from typing import Optional

from .protocol import S3ReaderConstructorProtocol
from .sequential import SequentialS3Reader
from .ranged import RangedS3Reader


class S3ReaderConstructor:
    """Constructor for creating partial(S3Reader) instances.

    Creates partial ``S3Reader`` instances that will be completed by ``S3Client`` with the
    remaining required parameters (e.g. ``bucket``, ``key``, ``get_object_info``, ``get_stream``).

    The constructor provides factory methods for different reader types:

    - ``sequential()``: Creates a constructor for sequential readers that buffer the entire object.
      Best for full reads and repeated access.
    - ``range_based()``: Creates a constructor for range-based readers that fetch specific byte ranges
      on-demand. Suitable for partial reads of large objects.

    Examples:
        For sequential reading (default)::

            reader_constructor = S3ReaderConstructor.sequential()

        For range-based reading::

            reader_constructor = S3ReaderConstructor.range_based()

        Using with ``S3MapDataset``::

            dataset = S3MapDataset.from_prefix(
                DATASET_URI,
                region=REGION,
                reader_constructor=S3ReaderConstructor.range_based()
            )
    """

    @staticmethod
    def sequential() -> S3ReaderConstructorProtocol:
        """Creates a constructor for sequential readers"""
        return partial(SequentialS3Reader)

    @staticmethod
    def range_based() -> S3ReaderConstructorProtocol:
        """Creates a constructor for range-based readers"""
        return partial(RangedS3Reader)

    @staticmethod
    def default() -> S3ReaderConstructorProtocol:
        """Returns the default reader constructor (sequential)"""
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
