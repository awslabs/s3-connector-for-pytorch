#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

"""
DCP-Optimized S3 Reader provides these 3 optimizations:
1. Selective data fetching with range coalescing to only fetch required byte ranges
2. Per-item buffer management to reduce buffer allocation costs
3. Eliminating buffer copy by storing S3 chunks as memoryview references

Data Flow Overview:
    DCP.load(model_state_dict, storage_reader=s3_storage_reader)
        -> read_metadata()                                          # reads .metadata file
        -> set_up_storage_reader(metadata)                          # populates storage_data
        -> prepare_local_plan(plan)                                 # (patched) sorts items, injects ranges to constructor
            -> DCPOptimizedConstructor.set_item_ranges_by_file()
        -> read_data(plan)                                          # per-file loop below
            -> DCPOptimizedS3Reader __init__
                -> _validate_and_coalesce_ranges()                  # validates and groups ItemRanges into RangeGroups
            -> DCPOptimizedS3Reader read()/readinto()
                -> _find_item_for_range()                        # updates _current_item
                -> [if new item] _get_item_buffer()                 # fetches item byte data
                    -> [if new RangeGroup] _get_stream_for_item()   # creates new stream before fetching byte data
                    -> 1: Handle leftover bytes from prev. chunk
                    -> 2: Skip gap data from coalescing
                    -> 3: Fetch remaining data from S3
                -> _ItemViewBuffer read()/readinto()                # returns data from buffer
"""

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

# Based on: throughput (~2500MB/s) × first-byte latency (~200ms) = ~500MB
DEFAULT_MAX_GAP_SIZE = 512 * 1024 * 1024
FIND_ITEM_ERROR_PREFIX = (
    "DCPOptimizedS3Reader only supports sequentially accessing provided ranges: "
)


@dataclass
class ItemRange:
    """Byte range for a single DCP ReadItem (tensor). Inclusive start, exclusive end."""

    start: int
    end: int


@dataclass
class RangeGroup:
    """Group of nearby ItemRanges that will share a single S3 range request.

    Created by coalescing ItemRanges with gaps <= max_gap_size in _validate_and_coalesce_ranges.
    One S3 stream will serve all items in the RangeGroup sequentially.
    """

    start: int  # First byte of the group (= first item's start)
    end: int  # Last byte of the group (= last item's end)
    item_ranges: List[ItemRange]  # Items within this group, in order


@dataclass
class _StreamState:
    """
    Tracks S3 stream position in current RangeGroup and buffered data between item reads.

    A single S3 stream may serve multiple items in a RangeGroup. This state tracks
        1. Where we are in that stream created for the current RangeGroup, and
        2. any leftover bytes from the previous chunk that belong to the next item.
    """

    stream: Optional[GetObjectStream] = None  # Current S3 stream for active RangeGroup
    stream_position: int = (
        -1
    )  # Current byte position in S3 stream; -1 as dummy init value
    leftover: Optional[memoryview] = None  # Unused bytes from end of last chunk read


# TODO: extend buffer for use in other S3Reader implementations after extensive testing
class _ItemViewBuffer:
    """
    A read-only buffer storing item data, in the form of multiple memoryview segments.

    Instead of copying S3 chunks into a growing BytesIO buffer, this class stores
    references to the original S3 chunks (typically 8MB parts) as memoryview segments.
    This allows us to reduce buffer allocation costs, saving time and memory.

    The buffer supports seek/read/readinto with logic that handles reads spanning
    multiple segments, similar to the file-access interface in io.BytesIO.
    """

    __slots__ = ("_segments", "_offsets", "_lengths", "_size", "_pos")

    def __init__(self) -> None:
        self._segments: List[memoryview] = []  # memoryview segments
        self._offsets: List[int] = []  # start offset (within the item) of each segment
        self._lengths: List[int] = (
            []
        )  # length of each segment (avoid recalculations from offset)
        self._size: int = 0  # total item length (sum of _lengths)
        self._pos: int = 0  # current read position within the item

    def append_view(self, view: memoryview) -> None:
        """Append a memoryview segment (ignored if empty)."""

        seg_len = len(view)
        if seg_len == 0:
            return
        self._segments.append(view)
        self._offsets.append(self._size)
        self._lengths.append(seg_len)
        self._size += seg_len

    def seek(self, offset: int, whence: int = SEEK_SET, /) -> int:
        assert isinstance(offset, int), f"integer expected, got {type(offset)!r}"

        if whence == SEEK_SET:
            new_pos = offset
        elif whence == SEEK_CUR:
            new_pos = self._pos + offset
        elif whence == SEEK_END:
            new_pos = self._size + offset
        else:
            raise ValueError("Seek must be passed SEEK_CUR, SEEK_SET, or SEEK_END")

        assert new_pos >= 0, f"negative seek value {new_pos}"

        # Seeking past EOF is allowed.
        self._pos = new_pos
        return self._pos

    def tell(self) -> int:
        """Return the current pos position (like BytesIO.tell)."""
        return self._pos

    def read(self, size: Optional[int] = None) -> bytes:
        """Returns byte copy of data from the buffer, using readinto() logic.

        Note that in DCP, only PyTorch's serialization.py:: _is_zipfile() magic
        number check (read(4)) uses read() instead of readinto().
        """

        # DCPOptimizedS3Reader doesn't allow full read, and doesn't use full reads on items either.
        assert size is not None, "Size cannot be None; full read is not supported"
        assert size >= 0, "Size cannot be negative; full read is not supported"

        # Fast path: PyTorch's serialization.py::_is_zipfile() reads first 4 bytes to check magic number.
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
        """Read into pre-allocated buffer, copying across segment boundaries as needed."""
        # Avoid creating new memoryview if input already is one
        dest = buf if isinstance(buf, memoryview) else memoryview(buf)
        assert not dest.readonly, "writable buffer required"

        dest_len = len(dest)
        size = self._size
        pos = self._pos

        if dest_len == 0 or pos >= size:
            return 0

        # Cache to avoid repeated attribute lookups in the loop
        segments = self._segments
        offsets = self._offsets
        lengths = self._lengths

        # Find segment containing pos with binary search
        # bisect_right gives insertion point where pos < offsets[i], -1 gives containing segment.
        # No caching optimisation, since torch.load jumps around (magic bytes, zip dir, tensor data)
        seg_idx = bisect.bisect_right(offsets, pos) - 1
        # Defensive clamp: shouldn't occur as _find_item_for_range makes sure pos >= 0
        if seg_idx < 0:
            seg_idx = 0

        written = 0
        bytes_to_read = min(dest_len, size - pos)

        # Copy from segments to dest, handling segment boundaries
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
    """S3 reader implementation optimized for PyTorch Distributed Checkpoint (DCP) loading.

    Provides up to 2x performance improvement over default sequential reader through:

        1. **Selective data fetching with range coalescing**: Uses byte range information from
        PyTorch's ``LoadPlan`` to only fetch required data. Groups nearby ranges within
        ``max_gap_size`` into single S3 streams to minimize first-byte latency while avoiding
        unnecessary data transfer.

        2. **Per-item buffer management**: Buffers per-item (per-tensor) instead of per-file. Each
        buffer stores only the required item's byte ranges and is discarded after PyTorch reads the
        item, which removes overhead of resizing large buffers and re-copying data repeatedly.

        3. **Eliminate buffer copy**: Custom ``_ItemViewBuffer`` stores S3 chunks as memoryview
        references instead of copying into BytesIO, avoiding allocation and copy overhead.

    **Requirements**:

    - DCP Loading - reader is only designed for usage via dcp_optimized reader_constructor for ``dcp.load()``
    - Pre-sorted list of item_ranges, injected automatically in ``prepare_local_plan``.
    - Sequential Access over exact item_ranges provided, also applied automatically by ``prepare_local_plan``

    **Usage**:
    Created automatically by ``DCPOptimizedConstructor`` when used with ``S3StorageReader`` and
    ``S3ReaderConstructor.dcp_optimized()``:

        reader_constructor = S3ReaderConstructor.dcp_optimized(max_gap_size=32*1024*1024)
        storage_reader = S3StorageReader(region, path, reader_constructor=reader_constructor)
        DCP.load(state_dict, storage_reader=storage_reader)

    **Error Handling**:
        Non-sequential access attempts raise ValueError.
    """

    def __init__(
        self,
        bucket: str,
        key: str,
        item_ranges: List[ItemRange],
        get_object_info: Callable[[], Union[ObjectInfo, HeadObjectResult]],
        get_stream: Callable[[Optional[int], Optional[int]], GetObjectStream],
        max_gap_size: Union[int, float] = DEFAULT_MAX_GAP_SIZE,
        # added float type to allow float("inf") / sys.maxsize for max_gap_size
    ):
        if not bucket:
            raise ValueError("Bucket should be specified")
        if not key:
            raise ValueError("Key should be specified")
        if not item_ranges:
            raise ValueError("item_ranges must be a non-empty List[ItemRange] object")
        if not isinstance(max_gap_size, (int, float)):
            raise TypeError(
                f"max_gap_size must be int or float, got {type(max_gap_size).__name__}"
            )
        if max_gap_size < 0:
            raise ValueError("max_gap_size must be non-negative")

        self._bucket = bucket
        self._key = key
        self._get_object_info = get_object_info
        self._get_stream = get_stream
        self._max_gap_size = max_gap_size
        self._closed = False

        # --- Range Processing ---

        # Filter zero-length ranges
        self._item_ranges: List[ItemRange] = [
            r for r in item_ranges if r.end != r.start
        ]
        if not self._item_ranges:
            raise ValueError("No non-empty ranges to read (all ranges were length 0)")

        # Coalesce nearby ranges into range groups that share S3 streams
        # _group_start_to_group: lookup dict for O(1) "is this item first in its group?" check
        self._group_start_to_group: Dict[int, RangeGroup] = {}
        self._range_groups: List[RangeGroup] = self._validate_and_coalesce_ranges(
            self._item_ranges, self._max_gap_size
        )

        # --- States ---

        # Stream state (stores S3 stream, position, and leftover bytes between item reads)
        self._stream_state: _StreamState = _StreamState()

        # Item buffer state
        self._item_iter: Iterator[ItemRange] = iter(
            self._item_ranges
        )  # sequential access
        self._current_item: ItemRange = next(self._item_iter)
        self._current_item_buffer: Optional[_ItemViewBuffer] = None

        # Current position in the overall S3 object
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
        self,
        ranges: List[ItemRange],
        max_gap_size: Union[int, float],
    ) -> List[RangeGroup]:
        """Validate ranges and coalesce nearby ranges into RangeGroups.

        - Validate: 1/ start<=end, 2/ non-negative, 3/ sorted by start position, 4/ non-overlapping
        - Coalesce: Group nearby ranges where gap <= max_gap_size into RangeGroup (one S3 stream).
        """
        if not ranges:
            return []

        groups: List[RangeGroup] = []
        items: List[ItemRange] = [ranges[0]]

        if not 0 <= ranges[0].start <= ranges[0].end:
            raise ValueError(f"Invalid range: {ranges[0].start}-{ranges[0].end}")
        for r in ranges[1:]:
            if r.end <= r.start:  # Empty ranges filtered out in __init__
                raise ValueError(f"Invalid range: {r.start}-{r.end}")
            if r.start < items[-1].end:
                if r.start < items[-1].start:
                    raise ValueError(
                        f"Unsorted ranges: {items[-1].start}-{items[-1].end} and {r.start}-{r.end}"
                    )
                else:
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

    def _find_item_for_range(self, start: int, end: int) -> ItemRange:
        """Find which item contains the requested range [start,end), enforcing sequential access.

        Returns current item if range is within it, and advances to / returns next item if the
        range is within the next item.

        Raises human-readable errors if the range is partially/fully outside of current or next items.
        """

        item = self._current_item

        # Check if requested range is within current item
        if item.start <= start and end <= item.end:
            return item

        # 1. If start < item.end and range not within current item, raise error
        # 2. Protection against reading 2nd item before first item by checking if buffer
        # contains any data (since we initialize with 1st item instead of None)
        if start < item.end or self._current_item_buffer is None:
            raise ValueError(
                f"{FIND_ITEM_ERROR_PREFIX}Range {start}-{end} not contained in "
                f"current item {item.start}-{item.end}"
            )

        # Advance to next item
        prev_item = item
        try:
            item = next(self._item_iter)
        except StopIteration:
            raise ValueError(
                f"{FIND_ITEM_ERROR_PREFIX}Range {start}-{end} not contained in last item "
                f"with range {prev_item.start}-{prev_item.end}"
            )

        # Check if requested range is within next item
        if item.start <= start and end <= item.end:
            return item

        raise ValueError(
            f"{FIND_ITEM_ERROR_PREFIX}Range {start}-{end} not contained in "
            f"current item {prev_item.start}-{prev_item.end} nor the "
            f"next item {item.start}-{item.end}."
        )

    def _get_stream_for_item(self, item: ItemRange) -> GetObjectStream:
        """Get or create S3 stream for the given item.

        Creates new stream if item is first in its RangeGroup, otherwise reuse stream.

        Each RangeGroup maps to a contiguous byte range in S3, and items within a
        RangeGroup are read sequentially from the same stream.

        Sequential access is already enforced in _find_item_for_range, which runs
        before _get_item_buffer (which calls this method). Reading the first item will
        trigger stream creation, and subsequent reads will simply reuse the stream.
        """

        # If item is the first item of a new group, create new stream
        if item.start in self._group_start_to_group:
            group = self._group_start_to_group[item.start]
            self._stream_state = _StreamState(
                stream=self._get_stream(group.start, group.end),
                stream_position=group.start,
                leftover=None,
            )
            assert self._stream_state.stream is not None
            return self._stream_state.stream

        # Otherwise, we're still in same group - reuse stream created when reading 1st item
        assert (  # Assert mainly serves for mypy checks.
            self._stream_state.stream is not None
        ), "No stream found for item; first item of its range group likely not read"
        return self._stream_state.stream

    def _get_item_buffer(self, item: ItemRange) -> _ItemViewBuffer:
        """Load entire item into a memoryview-segment buffer from existing stream.

        1. Handles leftover bytes from previous reads
        2. Skips gap data from coalescing within <=max_gap_size
        3. Fetches item data from S3 stream into buffer

        Returns buffer ready for read/readinto calls.
        """

        buffer = _ItemViewBuffer()

        # Get stream from the right RangeGroup for start_pos
        stream = self._get_stream_for_item(item)
        pos = self._stream_state.stream_position  # global offset in S3 object
        leftover = self._stream_state.leftover  # leftover from previous chunk
        bytes_left = item.end - item.start

        # --- Phase 1: Handle leftover bytes from previous chunk ---
        #
        # Leftover contains bytes from the end of the previous chunk (say 8MB) that weren't consumed.
        # The leftover always ends at a chunk boundary (of 8MB parts - assume 8MB from now for explanation)
        #
        # Two cases:
        #   A) item.start within leftover: extract needed portion, possibly
        #      i) skipping a prefix (gap data from coalescing ranges within max_gap_size), and/or
        #      ii) saving a suffix (next item's data) as new leftover
        #   B) item.start beyond leftover: discard all (gap data from coalescing ranges)
        #
        # Case A visualization (item starts within leftover):
        #
        #   8MB chunks: ...====|================================|====...
        #   leftover:                     |#####################|           (length: leftover_len)
        #                                 ^                     ^
        #                                pos             leftover_end_pos   (global position in object)
        #   Slice offsets:                |gap|used|new_leftover|           (gap/new_leftover can be empty)
        #                                     ^    ^
        #                              item.start item.end                  (global position in object)
        #                       start_in_leftover  end_in_leftover          (relative to leftover)
        #   Lengths:                          |<--------------->|
        #                                       available_bytes
        #                                     |<-->|
        #                                 bytes_to_extract
        #
        if leftover:
            leftover_len = len(leftover)
            leftover_end_pos = pos + leftover_len

            if pos <= item.start < leftover_end_pos:
                # Case A: Item starts within leftover data:
                # i) if there's gap data to skip, ignore it
                start_in_leftover = item.start - pos
                # ii) if more bytes than required, save suffix as new leftover
                available_bytes = leftover_len - start_in_leftover
                bytes_to_extract = min(bytes_left, available_bytes)
                end_in_leftover = start_in_leftover + bytes_to_extract

                # Extract needed portion to buffer, and update leftover
                buffer.append_view(leftover[start_in_leftover:end_in_leftover])
                bytes_left -= bytes_to_extract
                pos = item.start + bytes_to_extract
                leftover = (  # Update 'new_leftover'
                    leftover[end_in_leftover:]
                    if end_in_leftover < leftover_len
                    else None
                )
            elif leftover_end_pos <= item.start:
                # Case B: Item beyond leftover: discard leftover (it was gap data)
                pos += leftover_len
                leftover = None

        # --- Phase 2: Skip gap data (from coalescing) ---
        # Current state: pos is at chunk boundary of 8MB parts after any leftover is processed.
        # When ranges are coalesced (within max_gap_size), there may be gap data to skip.
        # So we iterate stream until chunk contains item.start.
        #
        # Two cases per chunk:
        #   A) Full chunk is gap: discard entirely, continue till pos >= item.start
        #   B) Boundary chunk: current chunk contains item.start
        #
        # Case B visualization (boundary chunk):
        #
        #   8MB chunks: ...====|================================|====...
        #   Fetched chunk:     |################################|           (length: chunk_len)
        #                      ^                                ^
        #                     pos                          pos+chunk_len
        #   Slice offsets:     | gap |   used   | new_leftover  |           (gap/new_leftover can be empty)
        #                            ^          ^
        #                     item.start     item.end                       (global position in object)
        #   Lengths:           |<--->|
        #                    skip_bytes
        #
        while pos < item.start:
            try:
                chunk = memoryview(next(stream))
            except StopIteration:
                break

            chunk_len = len(chunk)

            if pos + chunk_len <= item.start:
                # Entire chunk before item start - skip completely
                pos += chunk_len
                continue
            else:
                # Partial Skip - slice off unwanted part first
                skip_bytes = item.start - pos
                chunk = chunk[skip_bytes:]
                pos = item.start
                chunk_len -= skip_bytes

                # Now process boundary chunk
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

        # --- Phase 3: Fetch remaining item data ---
        # Current state: pos is at chunk boundary, and pos == item.start + [bytes extracted]
        # If bytes_left > 0, we still need more data for this item.
        #
        # Two cases per chunk:
        #   A) Full chunk needed: add all used bytes to buffer, continue if bytes_left > 0
        #   B) Partial chunk: item ends mid-chunk, add used bytes to buffer and update leftover
        #
        # Case B visualization (partial chunk):
        #
        #   8MB chunks: ...====|================================|====...
        #   Fetched chunk:     |################################|           (length: chunk_len)
        #                      ^                                ^
        #                     pos                          pos+chunk_len
        #   Slice offsets:     |  used  |      new_leftover     |
        #                      ^        ^
        #                     pos      item.end                             (global position in object)
        #   Lengths:           |<------>|
        #                      bytes_left                                   (bytes_left only if item ends here)
        #
        while bytes_left > 0:
            try:
                chunk = memoryview(next(stream))
            except StopIteration:
                break

            chunk_len = len(chunk)

            if chunk_len <= bytes_left:
                # Entire chunk needed - skip slicing
                buffer.append_view(chunk)
                bytes_left -= chunk_len
                pos += chunk_len
                leftover = None
            else:
                # Only part of chunk needed
                buffer.append_view(chunk[:bytes_left])
                leftover = chunk[bytes_left:]  # new_leftover
                pos += bytes_left
                bytes_left = 0
                break

        self._stream_state.stream_position = pos
        self._stream_state.leftover = leftover
        return buffer

    def read(self, size: Optional[int] = None) -> bytes:
        """
        Read up to size bytes from the current position.

        Supports backward seeking within the current item buffer, but forward-only
        access across DCP items (sequential item access required).

        Args:
            size (int): how many bytes to read.

        Returns:
            bytes: Bytes read from specified range.

        Raises:
            TypeError: If size is not an integer.
            ValueError: If position is outside valid DCP ranges, and if size is None or negative (full file reads not supported).
            S3Exception: An error occurred accessing S3.
        """
        if size is None:
            raise ValueError("Size cannot be None; full read not supported")
        if not isinstance(size, int):
            raise TypeError(f"argument should be integer or None, not {type(size)!r}")
        if size < 0:
            raise ValueError("Size cannot be negative; full read not supported")
        if size == 0:
            return b""

        item = self._find_item_for_range(self._position, self._position + size)

        # if item has been changed (or first item), then load new item to buffer
        if item is not self._current_item or self._current_item_buffer is None:
            self._current_item = item
            self._current_item_buffer = self._get_item_buffer(item)

        # Convert global position to item-relative offset for buffer seek
        local_pos_in_item_buffer = self._position - item.start
        self._current_item_buffer.seek(local_pos_in_item_buffer)
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
        item = self._find_item_for_range(self._position, self._position + len(buf))

        # if item has been changed (or first item), then load new item to buffer
        if item is not self._current_item or self._current_item_buffer is None:
            self._current_item = item
            self._current_item_buffer = self._get_item_buffer(item)

        # Convert global position to item-relative offset for buffer seek
        local_pos_in_item_buffer = self._position - item.start
        self._current_item_buffer.seek(local_pos_in_item_buffer)
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
        """
        if not isinstance(offset, int):
            raise TypeError(f"integer argument expected, got {type(offset)!r}")

        if whence == SEEK_SET:
            self._position = offset
        elif whence == SEEK_CUR:
            self._position += offset
        else:
            raise ValueError("whence must be SEEK_CUR or SEEK_SET integers")

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
        """
        Close the stream and release resources.
        """
        if not self._closed:
            self._closed = True
            self._stream_state.stream = None
            self._stream_state.leftover = None
            if self._current_item_buffer:
                self._current_item_buffer = None
