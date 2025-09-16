#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import os
import logging
from dataclasses import dataclass
from typing import List, Optional, Callable, Union, Dict
from io import SEEK_SET

from s3torchconnectorclient._mountpoint_s3_client import (
    ObjectInfo,
    GetObjectStream,
    HeadObjectResult,
)
from .s3reader import S3Reader
from .sequential import SequentialS3Reader

log = logging.getLogger(__name__)


@dataclass
class RangeRequest:
    start: int
    end: int
    request_id: Optional[str] = None


@dataclass
class RangeGroup:
    start: int
    end: int
    requests: List[RangeRequest]


class ListOfRangesS3Reader(S3Reader):
    """Optimized reader with pre-calculated request mapping and batch prefetch."""

    def __init__(
        self,
        bucket: str,
        key: str,
        ranges: List[RangeRequest],
        get_object_info: Callable[[], Union[ObjectInfo, HeadObjectResult]],
        get_stream: Callable[[Optional[int], Optional[int]], GetObjectStream],
        max_gap_size: int = 200 * 1024 * 1024,
        **kwargs,
    ):
        self._bucket = bucket
        self._key = key
        self._get_object_info = get_object_info
        self._get_stream = get_stream

        # Calculate range groups using coalescing logic
        self._range_groups = self._calculate_range_groups(ranges, max_gap_size)

        # Pre-create all readers and prefetch immediately
        # TODO - judge if this is beneficial or not.
        self._group_readers: Dict[int, SequentialS3Reader] = {}
        for i, group in enumerate(self._range_groups):
            reader = SequentialS3Reader(
                bucket=bucket,
                key=key,
                get_object_info=get_object_info,
                get_stream=get_stream,
                start_offset=group.start,
                end_offset=group.end,
            )
            reader.prefetch()  # Batch prefetch all ranges
            self._group_readers[i] = reader

        # Pre-calculate request-to-reader mapping
        self._request_to_reader: Dict[int, int] = {}
        for i, group in enumerate(self._range_groups):
            for request in group.requests:
                self._request_to_reader[request.start] = i

        self._current_position = 0

    @property
    def bucket(self) -> str:
        return self._bucket

    @property
    def key(self) -> str:
        return self._key

    def _calculate_range_groups(
        self, ranges: List[RangeRequest], max_gap_size: int
    ) -> List[RangeGroup]:
        """Coalescing logic - group ranges within max_gap_size."""
        # TODO: optimise this logic
        if not ranges:
            return []

        sorted_ranges = sorted(ranges, key=lambda r: r.start)
        groups = []
        current_group = [sorted_ranges[0]]

        for i in range(1, len(sorted_ranges)):
            prev_end = current_group[-1].end
            curr_start = sorted_ranges[i].start

            if curr_start - prev_end <= max_gap_size:
                current_group.append(sorted_ranges[i])
            else:
                groups.append(self._create_range_group(current_group))
                current_group = [sorted_ranges[i]]

        groups.append(self._create_range_group(current_group))
        return groups

    def _create_range_group(self, ranges: List[RangeRequest]) -> RangeGroup:
        """Create range group - always succeeds since we only use gap size."""
        # TODO remove min/max code by tracking incrementally in _calculate_range_groups
        # * (was kept since it's easier to understand and test)
        group_start = min(r.start for r in ranges)
        group_end = max(r.end for r in ranges)
        return RangeGroup(start=group_start, end=group_end, requests=ranges)

    def get_reader_for_request(
        self, request_start: int
    ) -> Optional[SequentialS3Reader]:
        """O(1) lookup using pre-calculated mapping."""
        reader_idx = self._request_to_reader.get(request_start)
        return self._group_readers.get(reader_idx) if reader_idx is not None else None

    def _find_reader_for_offset(self, offset: int) -> Optional[SequentialS3Reader]:
        """Find reader that contains the given offset."""
        # TODO: improve logic using binary search
        for reader in self._group_readers.values():
            if reader._start_offset <= offset < reader._end_offset:
                return reader
            elif reader._start_offset > offset:
                break  # Early termination since readers are ordered
        return None

    def seek(self, offset: int, whence: int = SEEK_SET, /) -> int:
        self._current_position = offset
        reader = self._find_reader_for_offset(offset)
        if not reader:
            return self._current_position
        reader.seek(offset, whence)

    def read(self, size: Optional[int] = None) -> bytes:
        reader = self._find_reader_for_offset(self._current_position)
        if not reader:
            return b""
        data = reader.read(size)
        self._current_position += len(data)
        return data

    def readinto(self, buf) -> int:
        reader = self._find_reader_for_offset(self._current_position)
        if not reader:
            return 0
        bytes_read = reader.readinto(buf)
        self._current_position += bytes_read
        return bytes_read

    def tell(self) -> int:
        return self._current_position

    def close(self) -> None:
        for reader in self._group_readers.values():
            reader.close()
