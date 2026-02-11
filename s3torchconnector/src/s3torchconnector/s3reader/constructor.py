#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import logging
import os
from functools import partial
from typing import TYPE_CHECKING, Optional, List, Dict, Union
from collections import defaultdict

from .s3reader import S3Reader
from .protocol import (
    S3ReaderConstructorProtocol,
    DCPS3ReaderConstructorProtocol,
)
from .sequential import SequentialS3Reader
from .ranged import RangedS3Reader
from .dcp_optimized import DCPOptimizedS3Reader, ItemRange, DEFAULT_MAX_GAP_SIZE

if TYPE_CHECKING:
    from torch.distributed.checkpoint.planner import ReadItem
    from torch.distributed.checkpoint.metadata import MetadataIndex
    from torch.distributed.checkpoint.filesystem import _StorageInfo

log = logging.getLogger(__name__)


class DCPOptimizedConstructor:
    """Constructor for DCPOptimizedS3Reader instances with range metadata injection.

    Created from S3ReaderConstructor, and used by S3StorageReader for PyTorch DCP.
    Requires range metadata from DCP load plans to function properly.

    Usage Flow:
        # User Setup
        reader_constructor = S3ReaderConstructor.dcp_optimized()  -> DCPOptimizedConstructor
        S3StorageReader(..., reader_constructor)                  -> stores in S3FileSystem

        # During DCP.load()
        S3StorageReader.prepare_local_plan(plan)
            -> set_item_ranges_by_file(plan.items, storage_data, path)
            -> builds _item_ranges_by_file: {s3_uri: [ItemRange, ...]}

        # During read_data(), per file
        S3FileSystem.create_stream(path, "rb")
            -> S3Client.get_object(bucket, key, reader_constructor)
                -> __call__(bucket, key, get_object_info, get_stream)
                    -> .metadata: SequentialS3Reader (no ranges available for .metadata file)
                    -> .distcp:   DCPOptimizedS3Reader(item_ranges) ranges from _item_ranges_by_file
    """

    def __init__(self, max_gap_size: Union[int, float] = DEFAULT_MAX_GAP_SIZE) -> None:

        if max_gap_size < 0:
            raise ValueError("max_gap_size must be non-negative")

        self._item_ranges_by_file: Dict[str, List[ItemRange]] = {}
        self._max_gap_size: Union[int, float] = max_gap_size

    def set_item_ranges_by_file(
        self,
        plan_items: "List[ReadItem]",
        storage_data: "Dict[MetadataIndex, _StorageInfo]",
        base_path: Union[str, os.PathLike],
    ) -> None:
        """Extract and store item ranges from DCP load plan.
        Called by S3StorageReader.prepare_local_plan() to inject range metadata.

        Note: This replaces any previously stored ranges (intentional for multi-call scenarios).
        """

        if not plan_items:
            # Empty plan: no reads will happen since no-op in FileSystemReader.read_data
            # To defend against multi-processing case where a rank doesn't need data.
            log.debug("Empty plan for this rank - no DCP item ranges will be created.")
            self._item_ranges_by_file = defaultdict(list)
            return

        if not storage_data:
            raise ValueError(
                "storage_data must not be empty; required to map ReadItems to file ranges."
            )

        # Map: full S3 URI -> item ranges
        self._item_ranges_by_file = defaultdict(list)
        for read_item in plan_items:
            item_md = storage_data[read_item.storage_index]
            s3_uri = os.path.join(base_path, item_md.relative_path)
            self._item_ranges_by_file[s3_uri].append(
                ItemRange(item_md.offset, item_md.offset + item_md.length)
            )

    def __call__(self, bucket: str, key: str, get_object_info, get_stream) -> S3Reader:
        """Match key to corresponding List[ItemRange] in _item_ranges_by_file.
        Fallback when reading .metadata file if ranges are unavailable.
        """

        # Return SequentialS3Reader for .metadata (hardcoded name in FileSystemReader.read_metadata),
        # since there are no ranges available for .metadata.
        # TODO: alternatively configure DCPOptimizedS3Reader to use full file range for .metadata.
        if key.endswith(".metadata"):
            log.debug(f"Reading .metadata file {key} with SequentialS3Reader")
            return SequentialS3Reader(bucket, key, get_object_info, get_stream)

        # Return DCPOptimizedS3Reader for .distcp files with corresponding item ranges for file uri
        s3_uri = f"s3://{bucket}/{key}"
        if s3_uri in self._item_ranges_by_file:
            return DCPOptimizedS3Reader(
                bucket,
                key,
                item_ranges=self._item_ranges_by_file[s3_uri],
                get_object_info=get_object_info,
                get_stream=get_stream,
                max_gap_size=self._max_gap_size,
            )

        # Error for other files; warn users in case they override prepare_local_plan behavior
        raise ValueError(
            f"No ranges found for {s3_uri}. Make sure range injection is used in S3StorageReader.prepare_local_plan."
        )


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
    def dcp_optimized(
        max_gap_size: Union[int, float] = DEFAULT_MAX_GAP_SIZE,
    ) -> DCPS3ReaderConstructorProtocol:
        """Creates a constructor for DCP-optimized readers for faster checkpoint loading.

        The DCP-optimized reader provides performance improvements for DCP reading through:

        - Selective data fetching with range coalescing to only fetch required byte ranges
        - Per-item buffer management to reduce buffer allocation costs
        - Eliminating buffer copy by storing S3 chunks as memoryview references

        Args:
            max_gap_size: Maximum gap size in bytes between ranges to coalesce into the same S3 read stream.
                Most users should use the default value.

                - Default: 32MB (``32 * 1024 * 1024``)
                - Use ``float("inf")`` to coalesce all ranges regardless of gaps
                - Use 0 to disable coalescing, which creates a new range-based stream for each gap

        Returns:
            DCPOptimizedConstructorProtocol:
                Constructor that creates DCPOptimizedS3Reader when ranges are available, falling back to
                SequentialS3Reader otherwise.

        Requirements:
            Should be used with S3StorageReader, in which ``prepare_local_plan()`` automatically handles:

            - Load ordering: Sorts items by storage offset for sequential access
            - Range injection: Provides byte ranges from DCP load plan to the reader

            Advanced users implementing custom readers must include these optimizations
            in their ``prepare_local_plan()``/``read_data()`` implementation to use the DCP-optimized reader.

        Example::

            reader_constructor = S3ReaderConstructor.dcp_optimized()
            storage_reader = S3StorageReader(region, path, reader_constructor=reader_constructor)
            DCP.load(state_dict, storage_reader=storage_reader)

        """
        return DCPOptimizedConstructor(max_gap_size=max_gap_size)

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

        if isinstance(constructor, DCPOptimizedConstructor):
            return "dcp_optimized"
        elif not isinstance(constructor, partial):
            return "unknown"
        elif constructor.func == RangedS3Reader:
            return "range_based"
        elif constructor.func == SequentialS3Reader:
            return "sequential"
        else:
            return "unknown"
