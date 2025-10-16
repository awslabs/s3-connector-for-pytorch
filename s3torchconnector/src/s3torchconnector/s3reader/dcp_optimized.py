#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import logging
from dataclasses import dataclass
from typing import List, Optional, Callable, Union, Dict, Iterator
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
        self._current_group_idx: int = 0

        # Per-group stream cache
        self._streams: Dict[int, Iterator[bytes]] = {}
        self._stream_positions: Dict[int, int] = {}
        self._stream_buffers: Dict[int, bytes] = {}

        self._position: int = 0

    @property
    def bucket(self) -> str:
        return self._bucket

    @property
    def key(self) -> str:
        return self._key

    def seekable(self) -> bool:
        """Not seekable — torch/distributed/checkpoint/filesystem.py will use read() instead of readinto()."""
        return False

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

    def _get_stream_for_group(self, idx: int) -> Iterator[bytes]:
        """
        Returns a cached iterator for the given range group,
        or creates a new one if not present.
        """
        if idx not in self._streams:
            group = self._range_groups[idx]
            stream = self._get_stream(group.start, group.end)
            self._streams[idx] = stream
            self._stream_positions[idx] = group.start
            self._stream_buffers[idx] = b""
        return self._streams[idx]

    def read(self, size: Optional[int] = None) -> bytes:
        """Reads up to `size` bytes sequentially across grouped ranges."""
        if not size or size <= 0:
            return b""

        pos = self._position

        # Find group (with cache)
        if (
            self._current_group_idx < len(self._range_groups)
            and self._range_groups[self._current_group_idx].start
            <= pos
            < self._range_groups[self._current_group_idx].end
        ):
            group_idx = self._current_group_idx
        else:
            # Search for matching group
            for i, g in enumerate(self._range_groups):
                if g.start <= pos < g.end:
                    group_idx = i
                    self._current_group_idx = group_idx
                    break
            else:
                return b""

        stream = self._get_stream_for_group(group_idx)

        current_pos = self._stream_positions[group_idx]
        buffer = self._stream_buffers[group_idx]
        remaining = size
        chunks: List[bytes] = []

        # 1. Serve from buffered leftover bytes
        if buffer and current_pos <= pos < current_pos + len(buffer):
            offset = pos - current_pos
            end = offset + min(remaining, len(buffer) - offset)
            chunks.append(buffer[offset:end])
            remaining -= end - offset
            current_pos = pos + (end - offset)
            self._stream_buffers[group_idx] = buffer[end:] if end < len(buffer) else b""

        # 2. Read more data from S3 stream
        while remaining > 0:
            try:
                chunk = next(stream)
            except StopIteration:
                break

            # Skip ahead if behind target
            if current_pos < pos:
                skip = min(pos - current_pos, len(chunk))
                chunk = chunk[skip:]
                current_pos += skip

            # Take needed part of chunk
            take = min(len(chunk), remaining)
            chunks.append(chunk[:take])
            remaining -= take
            current_pos += take

            # Save leftover bytes
            if take < len(chunk):
                self._stream_buffers[group_idx] = chunk[take:]
                break

        self._stream_positions[group_idx] = current_pos
        self._position = pos + (size - remaining)
        return b"".join(chunks)

    def seek(self, offset: int, whence: int = SEEK_SET, /) -> int:
        if whence == SEEK_SET:
            self._position = offset
        elif whence == SEEK_CUR:
            self._position += offset
        return self._position

    def readinto(self, buf) -> int:
        data = self.read(len(buf))
        n = len(data)
        buf[:n] = data
        return n

    def tell(self) -> int:
        return self._position

    def close(self) -> None:
        self._streams.clear()
        self._stream_positions.clear()
        self._stream_buffers.clear()
