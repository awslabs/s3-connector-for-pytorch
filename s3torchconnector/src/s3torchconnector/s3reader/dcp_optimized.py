#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import io
import logging
from dataclasses import dataclass
from typing import List, Optional, Callable, Union, Iterator
from io import SEEK_SET, SEEK_CUR

from s3torchconnectorclient._mountpoint_s3_client import (
    ObjectInfo,
    GetObjectStream,
    HeadObjectResult,
)
from .s3reader import S3Reader

log = logging.getLogger(__name__)

DEFAULT_MAX_GAP_SIZE = 200 * 1024 * 1024  # 200MB


@dataclass
class RangeRequest:
    """Singular range request; Inclusive start, exclusive end"""

    start: int
    end: int
    request_id: Optional[str] = None


@dataclass
class RangeGroup:
    start: int
    end: int
    requests: List[RangeRequest]


# TODO: Update docstrings to emphasise this requires Load Ordering in prepare_local_plan
class DCPOptimizedS3Reader(S3Reader):
    """Optimized reader with pre-calculated request mapping and batch prefetch."""

    def __init__(
        self,
        bucket: str,
        key: str,
        ranges: List[RangeRequest],
        get_object_info: Callable[[], Union[ObjectInfo, HeadObjectResult]],
        get_stream: Callable[[Optional[int], Optional[int]], GetObjectStream],
        max_gap_size: int = DEFAULT_MAX_GAP_SIZE,
    ):
        self._bucket = bucket
        self._key = key
        self._get_object_info = get_object_info
        self._get_stream = get_stream

        # Calculate range groups using coalescing logic
        if max_gap_size < 0:
            raise ValueError("max_gap_size must be non-negative")
        self._max_gap_size = max_gap_size
        self._range_groups = self._coalesce_ranges(ranges, self._max_gap_size)

        # Single active stream state
        self._gidx: int = 0  # current group index
        self._stream: Optional[Iterator[bytes]] = (
            None  # iterator over bytes for current group
        )
        self._stream_pos: int = 0  # absolute position at head of stream
        self._leftover: bytes = b""  # unconsumed tail of last chunk

        # Item-based buffering for seekable support
        self._item_ranges = sorted(ranges, key=lambda r: r.start)
        self._current_item_idx: int = 0
        self._current_item_buffer: Optional[io.BytesIO] = None

        self._position: int = 0

    @property
    def bucket(self) -> str:
        return self._bucket

    @property
    def key(self) -> str:
        return self._key

    def _coalesce_ranges(
        self, ranges: List[RangeRequest], max_gap_size: int
    ) -> List[RangeGroup]:
        """Coalescing nearby byte ranges within max_gap_size."""
        if not ranges:
            return []

        # TODO: could be pre-sorted in prepare_local_plan (small optimisation)
        ranges = sorted(ranges, key=lambda r: r.start)
        groups: List[RangeGroup] = []
        current = [ranges[0]]

        for r in ranges[1:]:
            if r.start - current[-1].end <= max_gap_size:
                current.append(r)
            else:
                groups.append(RangeGroup(current[0].start, current[-1].end, current))
                current = [r]

        groups.append(RangeGroup(current[0].start, current[-1].end, current))
        return groups

    def _find_item_for_position(self, pos: int) -> Optional[int]:
        """Find which item contains the given position with fast path optimization."""
        # Check current item first
        if (
            self._current_item_idx < len(self._item_ranges)
            and self._item_ranges[self._current_item_idx].start
            <= pos
            < self._item_ranges[self._current_item_idx].end
        ):
            return self._current_item_idx

        # Check next item (load ordering assumption)
        next_idx = self._current_item_idx + 1
        if (
            next_idx < len(self._item_ranges)
            and self._item_ranges[next_idx].start
            <= pos
            < self._item_ranges[next_idx].end
        ):
            return next_idx

        # Fallback: linear scan (should be rare)
        for i, item_range in enumerate(self._item_ranges):
            if item_range.start <= pos < item_range.end:
                return i
        return None

    def _load_item_buffer(self, item_idx: int) -> None:
        """Load entire item into buffer using streaming approach."""
        item = self._item_ranges[item_idx]
        data = self._stream_range_data(item.start, item.end)
        self._current_item_buffer = io.BytesIO(data)
        self._current_item_idx = item_idx

    def _stream_range_data(self, start_pos: int, end_pos: int) -> bytes:
        """Read required range from the active stream."""
        if start_pos >= end_pos:
            return b""

        size = end_pos - start_pos

        # Find group and ensure we have the right stream
        if (
            self._gidx < len(self._range_groups)
            and self._range_groups[self._gidx].start
            <= start_pos
            < self._range_groups[self._gidx].end
        ):
            group_idx = self._gidx
        else:
            for i, g in enumerate(self._range_groups):
                if g.start <= start_pos < g.end:
                    group_idx = i
                    self._gidx = i
                    break
            else:  # executes if the for loop completes witout break
                return b""

        # Ensure stream exists for current group
        if self._stream is None:
            g = self._range_groups[group_idx]
            self._stream = self._get_stream(g.start, g.end)
            self._stream_pos = g.start
            self._leftover = b""

        current_pos = self._stream_pos
        buffer = self._leftover
        remaining = size
        chunks: List[bytes] = []

        # 1. Serve from buffered leftover bytes
        if buffer and current_pos <= start_pos < current_pos + len(buffer):
            offset = start_pos - current_pos
            end = offset + min(remaining, len(buffer) - offset)
            chunks.append(buffer[offset:end])
            remaining -= end - offset
            current_pos = start_pos + (end - offset)
            buffer = buffer[end:] if end < len(buffer) else b""

        # 2. Read more data from S3 stream
        while remaining > 0:
            try:
                chunk = next(self._stream)
            except StopIteration:
                break

            # Skip ahead if behind target
            if current_pos < start_pos:
                skip = min(start_pos - current_pos, len(chunk))
                chunk = chunk[skip:]
                current_pos += skip

            # Take needed part of chunk
            take = min(len(chunk), remaining)
            chunks.append(chunk[:take])
            remaining -= take
            current_pos += take

            # Save leftover bytes
            if take < len(chunk):
                buffer = chunk[take:]
                break

        self._stream_pos = current_pos
        self._leftover = buffer
        return b"".join(chunks)

    def read(self, size: Optional[int] = None) -> bytes:
        if size is not None and size <= 0:
            return b""

        item_idx = self._find_item_for_position(self._position)
        if item_idx is None:
            return b""

        if item_idx != self._current_item_idx or self._current_item_buffer is None:
            self._load_item_buffer(item_idx)

        item_range = self._item_ranges[self._current_item_idx]
        local_pos = self._position - item_range.start

        assert self._current_item_buffer is not None
        self._current_item_buffer.seek(local_pos)
        data = self._current_item_buffer.read(size)
        self._position += len(data)
        return data

    def seek(self, offset: int, whence: int = SEEK_SET, /) -> int:
        if whence == SEEK_SET:
            self._position = offset
        elif whence == SEEK_CUR:
            self._position += offset
        return self._position

    def readinto(self, buf) -> int:
        item_idx = self._find_item_for_position(self._position)
        if item_idx is None:
            return 0

        if item_idx != self._current_item_idx or self._current_item_buffer is None:
            self._load_item_buffer(item_idx)

        item_range = self._item_ranges[self._current_item_idx]
        local_pos = self._position - item_range.start

        assert self._current_item_buffer is not None
        self._current_item_buffer.seek(local_pos)
        bytes_read = self._current_item_buffer.readinto(buf)
        self._position += bytes_read
        return bytes_read

    def tell(self) -> int:
        return self._position

    def close(self) -> None:
        self._stream = None
        self._leftover = b""
        if self._current_item_buffer:
            self._current_item_buffer.close()
            self._current_item_buffer = None
