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

DEFAULT_MAX_GAP_SIZE = 32 * 1024 * 1024  # TODO tune this default


# TODO: check if we can reuse dcp planner ReadItem instead
@dataclass
class ItemRange:
    """Byte range for a ReadItem; Inclusive start, exclusive end"""

    start: int
    end: int


@dataclass
class RangeGroup:
    start: int
    end: int
    item_ranges: List[ItemRange]


class DCPOptimizedS3Reader(S3Reader):
    """
    This reader optimizes PyTorch Distributed Checkpoint (DCP) partial loading by
        1. exploiting sequential access patterns to avoid BytesIO buffer copy, and
        2. only fetching required byte ranges instead of entire objects.

    REQUIRES:
    - DCP Loading - reader is only designed for usage via dcp_optimized reader_constructor for dcp.load()
    - Load Ordering, applied automatically prepare_local_plan, to ensure sequential access patterns.
    - item_ranges provided (List[ItemRange]) must be pre-sorted - also applied in prepare_local_plan.
    - Only supports sequentially reading exact item_ranges provided - otherwise would result in errors.
    Non-sequential access will result in errors.

    """

    def __init__(
        self,
        bucket: str,
        key: str,
        item_ranges: List[ItemRange],
        get_object_info: Callable[[], Union[ObjectInfo, HeadObjectResult]],
        get_stream: Callable[[Optional[int], Optional[int]], GetObjectStream],
        max_gap_size: int = DEFAULT_MAX_GAP_SIZE,
    ):
        self._bucket = bucket
        self._key = key
        self._get_object_info = get_object_info
        self._get_stream = get_stream
        self._max_gap_size = max_gap_size

        if not item_ranges:
            raise ValueError("ranges must be non-empty List[ItemRange] object")
        if max_gap_size < 0:
            raise ValueError("max_gap_size must be non-negative")

        # Coalesce ranges into range groups
        # TODO: add test/check that unsorted ranges would be detected and results in error
        self._item_ranges: List[ItemRange] = item_ranges
        self._range_groups: List[RangeGroup] = self._validate_and_coalesce_ranges(
            self._item_ranges, self._max_gap_size
        )

        # Stream state
        self._group_iter: Iterator[RangeGroup] = iter(self._range_groups)
        self._current_group: RangeGroup = next(self._group_iter)
        self._stream: Optional[GetObjectStream] = None
        self._stream_pos: int = (
            self._current_group.start
        )  # absolute position at head of stream
        self._leftover: bytes = b""

        # Item buffer state
        self._item_iter: Iterator[ItemRange] = iter(self._item_ranges)
        self._current_item: ItemRange = next(self._item_iter)
        self._current_item_buffer: Optional[io.BytesIO] = None

        self._position: int = 0

    @property
    def bucket(self) -> str:
        return self._bucket

    @property
    def key(self) -> str:
        return self._key

    def _validate_and_coalesce_ranges(
        self, ranges: List[ItemRange], max_gap_size: int
    ) -> List[RangeGroup]:
        """Coalescing nearby byte ranges within max_gap_size."""
        if not ranges:
            return []

        groups: List[RangeGroup] = []
        items: List[ItemRange] = [ranges[0]]

        # TODO: Could this validation be done in constructor.py instead?
        if ranges[0].start < 0 or ranges[0].end < ranges[0].start:
            raise ValueError(f"Invalid range: {ranges[0].start}-{ranges[0].end}")
        for r in ranges[1:]:
            if r.end < r.start:  # Allow empty range
                raise ValueError(f"Invalid range: {r.start}-{r.end}")
            if r.start < items[-1].end:
                raise ValueError(
                    f"Overlapping ranges: {items[-1].start}-{items[-1].end} and {r.start}-{r.end}"
                )
            # Coalesce or create new group
            if r.start - items[-1].end <= max_gap_size:
                items.append(r)
            else:
                groups.append(RangeGroup(items[0].start, items[-1].end, items))
                items = [r]

        groups.append(RangeGroup(items[0].start, items[-1].end, items))
        return groups

    def _find_item_for_position(self, pos: int) -> ItemRange:
        """Find which item contains the given position with fast path optimization."""

        # Check current
        if self._current_item.start <= pos < self._current_item.end:
            return self._current_item

        # Try next item
        try:
            next_item = next(self._item_iter)
            if next_item.start <= pos < next_item.end:
                return next_item
        except StopIteration:
            raise ValueError(f"Position {pos} beyond all ranges")

        # Error detected - construct and raise human-readable error message
        curr_item = self._current_item
        direction = "before" if pos < curr_item.start else "beyond"
        range_info = f"current range {curr_item.start}-{curr_item.end}"
        raise ValueError(f"Position {pos} {direction} {range_info}")

    def _get_stream_for_item(self, item: ItemRange) -> None:
        """Find which RangeGroup contains the given position."""

        # Check current stream
        group = self._current_group
        if group.start <= item.start <= item.end <= group.end:
            if self._stream is None:
                self._stream = self._get_stream(group.start, group.end)
                self._stream_pos = group.start
                self._leftover = b""
            return

        # Try next stream
        try:
            group = next(self._group_iter)
            if group.start <= item.start <= item.end <= group.end:
                self._current_group = group
                self._stream = self._get_stream(group.start, group.end)
                self._stream_pos = group.start
                self._leftover = b""
                return
        except StopIteration:
            raise ValueError(f"Item {item.start}-{item.end} beyond all groups")

        # Error detected - construct and raise human-readable error message
        curr_group = self._current_group
        direction = "before" if item.start < curr_group.start else "beyond"
        group_info = f"current group {curr_group.start}-{curr_group.end}"
        raise ValueError(f"Item {item.start}-{item.end} {direction} {group_info}")

    def _load_item_buffer(self, item: ItemRange) -> None:
        """Load entire item into BytesIO buffer from existing stream."""

        if item.start >= item.end:
            self._current_item_buffer = io.BytesIO(b"")
            return

        # Get stream from the right RangeGroup for start_pos
        self._get_stream_for_item(item)
        assert self._stream is not None

        current_pos = self._stream_pos
        buffer = self._leftover
        remaining = item.end - item.start
        chunks: List[bytes] = []

        # TODO: check BytesIO.write() vs b"".join() + BytesIO(data)

        # 1. Serve from buffered leftover bytes
        if buffer and current_pos <= item.start < current_pos + len(buffer):
            offset = item.start - current_pos
            take = min(remaining, len(buffer) - offset)
            chunks.append(buffer[offset : offset + take])
            remaining -= take
            current_pos = item.start + take
            buffer = buffer[offset + take :] if offset + take < len(buffer) else b""

        # 2. Read more data from S3 stream
        while remaining > 0:
            try:
                chunk = next(self._stream)
            except StopIteration:
                break

            # Skip ahead if behind target
            if current_pos < item.start:
                skip = min(item.start - current_pos, len(chunk))
                chunk = chunk[skip:]
                current_pos += skip

            # Take needed part of chunk
            if len(chunk) <= remaining:
                # Entire chunk needed - skip slicing
                chunks.append(chunk)
                remaining -= len(chunk)
                current_pos += len(chunk)
            else:
                # Only part of chunk needed
                chunks.append(chunk[:remaining])
                buffer = chunk[remaining:]
                current_pos += remaining
                remaining = 0
                break

        self._stream_pos = current_pos
        self._leftover = buffer
        data = b"".join(chunks)
        self._current_item_buffer = io.BytesIO(data)

    def read(self, size: Optional[int] = None) -> bytes:
        """
        Read up to size bytes from the current position.

        Supports backward seeking within the current item buffer, but forward-only
        access across DCP items (sequential item access required).

        Args:
            size (int | None): how many bytes to read.

        Returns:
            bytes: Bytes read from specified range.

        Raises:
            NotImplementedError: If size is None or negative (full file reads not supported).
            TypeError: If size is not an integer.
            ValueError: If position is outside valid DCP ranges.
            S3Exception: An error occurred accessing S3.
        """
        if size is None or size < 0:
            raise NotImplementedError(
                "Size cannot be negative, full read is not supported."
            )
        if not isinstance(size, int):
            raise TypeError(f"argument should be integer or None, not {type(size)!r}")
        if size == 0:
            return b""

        item = self._find_item_for_position(self._position)

        if item is not self._current_item or self._current_item_buffer is None:
            self._current_item = item
            self._load_item_buffer(item)

        local_pos = self._position - item.start

        assert self._current_item_buffer is not None
        self._current_item_buffer.seek(local_pos)
        data = self._current_item_buffer.read(size)
        self._position += len(data)
        return data

    def readinto(self, buf) -> int:
        """
        Read up to len(buf) bytes into a pre-allocated, writable bytes-like object buf.
        Return the number of bytes read. If no bytes are available, zero is returned.

        Args:
            buf : writable bytes-like object

        Returns:
            int : number of bytes read or zero, if no bytes available

        Raises:
            ValueError: If position is outside valid DCP ranges.
            TypeError: If buf is not writable.
            S3Exception: An error occurred accessing S3.
        """

        # TODO: remove view = memoryview(buf) for performance or simpler checks
        try:
            view = memoryview(buf)
            if view.readonly:
                raise TypeError(
                    f"argument must be a writable bytes-like object, not {type(buf).__name__}"
                )
        except TypeError:
            raise TypeError(
                f"argument must be a writable bytes-like object, not {type(buf).__name__}"
            )

        item = self._find_item_for_position(self._position)

        if item is not self._current_item or self._current_item_buffer is None:
            self._current_item = item
            self._load_item_buffer(item)

        local_pos = self._position - item.start

        assert self._current_item_buffer is not None
        self._current_item_buffer.seek(local_pos)
        bytes_read = self._current_item_buffer.readinto(buf)
        self._position += bytes_read
        return bytes_read

    def seek(self, offset: int, whence: int = SEEK_SET, /) -> int:
        """
        Change position within DCP ranges, interpreted relative to whence.

        Supports arbitrary seeking within current item buffer, but only forward
        sequential access across DCP items (cannot seek back to previous items).

        Args:
            offset (int): How many bytes to seek relative to whence.
            whence (int): One of SEEK_SET, and SEEK_CUR. SEEK_END not supported. Default: SEEK_SET.

        Returns:
            int: Current position of the stream

        Raises:
            TypeError: If whence is not SEEK_SET or SEEK_CUR.
            ValueError: If seeking to negative position or accessing previous items.
            TypeError: If whence is not SEEK_SET or SEEK_CUR.
        """
        if not isinstance(offset, int):
            raise TypeError(f"integer argument expected, got {type(offset)!r}")

        if whence == SEEK_SET:
            self._position = offset
        elif whence == SEEK_CUR:
            self._position += offset
        else:
            raise ValueError("Seek must be passed io SEEK_CUR or SEEK_SET integers")

        if self._position < 0:
            raise ValueError(f"negative seek value {self._position}")

        return self._position

    def tell(self) -> int:
        """
        Returns:
            int: Current absolute position in the object.
        """
        return self._position

    def close(self) -> None:
        self._stream = None
        self._leftover = b""
        if self._current_item_buffer:
            self._current_item_buffer.close()
            self._current_item_buffer = None
