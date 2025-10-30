#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import bisect
import logging
from dataclasses import dataclass
from typing import List, Optional, Callable, Union, Iterator, Dict, cast
from io import SEEK_SET, SEEK_CUR, SEEK_END

from s3torchconnectorclient._mountpoint_s3_client import (
    ObjectInfo,
    GetObjectStream,
    HeadObjectResult,
)
from .s3reader import S3Reader

log = logging.getLogger(__name__)

DEFAULT_MAX_GAP_SIZE = 32 * 1024 * 1024  # TODO tune this default


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


# TODO: extend buffer for use in other S3Reader implementations after extensive testing
class _ItemViewBuffer:
    """
    A tiny, zero-copy, read-only buffer built from multiple memoryview segments.
    Replaces io.BytesIO which involved extra copies for creation and buffer growth.
    """

    __slots__ = ("_segments", "_offsets", "_lengths", "_size", "_pos", "_closed")

    def __init__(self) -> None:
        self._segments: List[memoryview] = []  # memoryview segments
        self._offsets: List[int] = []  # start offset (within the item) of each segment
        self._lengths: List[int] = []  # length of each segment
        self._size: int = 0  # total item length (sum of _lengths)
        self._pos: int = 0  # current read position within the item
        self._closed: bool = False

    def append_view(self, view: memoryview) -> None:
        """Append a memoryview segment (ignored if empty)."""
        # assert so it disappears with -O
        assert not self._closed, "Buffer is closed"

        seg_len = len(view)
        if seg_len == 0:
            return
        self._segments.append(view)
        self._offsets.append(self._size)
        self._lengths.append(seg_len)
        self._size += seg_len

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._segments.clear()

    def seek(self, offset: int, whence: int = SEEK_SET, /) -> int:
        assert isinstance(offset, int), f"integer expected, got {type(offset)!r}"

        if whence == SEEK_SET:
            new_pos = offset
        elif whence == SEEK_CUR:
            new_pos = self._pos + offset
        elif whence == SEEK_END:
            new_pos = self._size + offset
        else:
            raise ValueError(
                "Seek must be passed io SEEK_CUR, SEEK_SET, or SEEK_END integers"
            )

        assert new_pos >= 0, f"negative seek value {new_pos}"

        # Seeking past EOF is allowed.
        self._pos = new_pos
        return self._pos

    def tell(self) -> int:
        """Return the current pos position (like BytesIO.tell)."""
        return self._pos

    def read(self, size: Optional[int] = None) -> bytes:
        assert size is not None, "Size cannot be None; full read is not supported"
        assert size >= 0, "Size cannot be negative; full read is not supported"

        # Fast path for read(4) at pos=0 (Optimizes pytorch/torch/serialization.py _is_zipfile())
        if size == 4 and self._pos == 0 and self._lengths and self._lengths[0] >= 4:
            self._pos = 4
            # TODO: eliminating bytes() conversion can save ~3% time? Requires interface changes.
            return bytes(self._segments[0][:4])

        if size == 0:
            return b""

        # Pass implementation to readinto()
        out = bytearray(size)
        n = self.readinto(out)
        return bytes(out) if n == size else memoryview(out)[:n].tobytes()

    def readinto(self, buf) -> int:
        # TODO: Check if we really need to wrap with memoryview
        dest = buf if isinstance(buf, memoryview) else memoryview(buf)
        assert not dest.readonly, "writable buffer required"

        dest_len = len(dest)
        size = self._size
        pos = self._pos

        if dest_len == 0 or pos >= size:
            return 0

        # Cache to avoid repeated attribute calls
        segments = self._segments
        offsets = self._offsets
        lengths = self._lengths

        # Starting segment idx: last i where _offsets[i] <= _pos
        seg_idx = bisect.bisect_right(offsets, pos) - 1
        if seg_idx < 0:
            seg_idx = 0

        written = 0
        bytes_to_read = min(dest_len, size - pos)

        # Copy from segments to dest
        while written < bytes_to_read:
            seg_start = offsets[seg_idx]
            seg_len = lengths[seg_idx]
            seg = segments[seg_idx]

            # Account for first chunk when pos > seg_start
            offset_in_seg = pos - seg_start

            # Account for last chunk when bytes_to_read < seg_len
            available_in_seg = seg_len - offset_in_seg
            bytes_left_to_read = bytes_to_read - written
            copy_size = min(bytes_left_to_read, available_in_seg)

            dest[written : written + copy_size] = seg[
                offset_in_seg : offset_in_seg + copy_size
            ]

            written += copy_size
            pos += copy_size
            seg_idx += 1

        self._pos += written
        return written


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
        self._closed = False

        if not item_ranges:
            raise ValueError("ranges must be non-empty List[ItemRange] object")
        if max_gap_size < 0:
            raise ValueError("max_gap_size must be non-negative")

        # Coalesce ranges into range groups
        # TODO: add test/check that unsorted ranges would be detected and results in error
        self._item_ranges: List[ItemRange] = item_ranges
        self._group_start_to_group: Dict[int, RangeGroup] = (
            {}
        )  # Group lookup using group start offsets
        self._range_groups: List[RangeGroup] = self._validate_and_coalesce_ranges(
            self._item_ranges, self._max_gap_size
        )

        # Stream state
        self._stream: Optional[GetObjectStream] = None
        self._stream_pos: int = -1  # position at head of stream - dummy int
        self._leftover: Optional[memoryview] = None

        # Item buffer state
        self._item_iter: Iterator[ItemRange] = iter(self._item_ranges)
        self._current_item: ItemRange = next(self._item_iter)
        self._current_item_buffer: Optional[_ItemViewBuffer] = None

        self._position: int = 0

    @property
    def bucket(self) -> str:
        return self._bucket

    @property
    def key(self) -> str:
        return self._key

    @property
    def closed(self) -> bool:
        """
        Returns:
            bool: Return whether the object is closed.
        """
        return self._closed

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
                group = RangeGroup(items[0].start, items[-1].end, items)
                groups.append(group)
                self._group_start_to_group[items[0].start] = group
                items = [r]

        final_group = RangeGroup(items[0].start, items[-1].end, items)
        self._group_start_to_group[items[0].start] = final_group
        groups.append(final_group)
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

    def _get_stream_for_item(self, item: ItemRange) -> GetObjectStream:
        """Find which RangeGroup contains the given position."""

        # Assuming stream exists if item is not at the start of any groups
        if not item.start in self._group_start_to_group:
            # 1st item always in _group_start_to_group - cast to reduce assert calls
            return cast(GetObjectStream, self._stream)

        group = self._group_start_to_group[item.start]
        self._stream = self._get_stream(group.start, group.end)
        self._stream_pos = group.start
        self._leftover = None
        return self._stream

    def _get_item_buffer(self, item: ItemRange) -> _ItemViewBuffer:
        """Load entire item into a memoryview-segment buffer from existing stream."""

        buffer = _ItemViewBuffer()
        if item.start >= item.end:
            return buffer

        # Get stream from the right RangeGroup for start_pos
        stream = self._get_stream_for_item(item)
        pos = self._stream_pos  # local copy
        leftover = self._leftover  # local copy
        bytes_left = item.end - item.start

        # 1. Read from leftover bytes if available and needed
        if leftover:
            lv_len = len(leftover)
            lv_end = pos + lv_len

            if pos <= item.start < lv_end:
                # Item starts within leftover data
                start = item.start - pos
                available_bytes = lv_len - start
                size = min(bytes_left, available_bytes)
                end = start + size

                # Extract needed portion
                buffer.append_view(leftover[start:end])
                bytes_left -= size
                pos = item.start + size
                leftover = leftover[end:] if end < lv_len else None
            elif item.start >= lv_end:
                # Item beyond leftover: advance pos to end of leftover
                pos += lv_len
                leftover = None

        # 2. Read more data from S3 stream
        while bytes_left > 0:
            try:
                chunk = memoryview(next(stream))
            except StopIteration:
                break

            chunk_len = len(chunk)

            # TODO: separate skip part and take part for clearer logic
            # Skip past unwanted data (due to coalescing)
            if pos < item.start:
                skip_bytes = min(item.start - pos, chunk_len)
                chunk = chunk[skip_bytes:]
                pos += skip_bytes
                chunk_len -= skip_bytes
                if chunk_len == 0:
                    continue

            # Take needed part of chunk
            if chunk_len <= bytes_left:
                # Entire chunk needed - skip slicing
                buffer.append_view(chunk)
                bytes_left -= chunk_len
                pos += chunk_len
                leftover = None
            else:
                # Only part of chunk needed
                buffer.append_view(chunk[:bytes_left])
                leftover = chunk[bytes_left:]
                pos += bytes_left
                bytes_left = 0
                break

        self._stream_pos = pos
        self._leftover = leftover
        return buffer

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
            self._current_item_buffer = self._get_item_buffer(item)

        # TODO: Can handle offset in _ItemViewBuffer instead
        local_pos = self._position - item.start

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

        item = self._find_item_for_position(self._position)

        if item is not self._current_item or self._current_item_buffer is None:
            self._current_item = item
            self._current_item_buffer = self._get_item_buffer(item)

        local_pos = self._position - item.start

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
        if not self._closed:
            self._closed = True
            self._stream = None
            self._leftover = None
            if self._current_item_buffer:
                self._current_item_buffer.close()
                self._current_item_buffer = None
