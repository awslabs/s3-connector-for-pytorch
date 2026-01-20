#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import re
import sys
from io import BytesIO, SEEK_SET, SEEK_CUR, SEEK_END
from typing import List, Tuple, Optional

import pytest
from hypothesis import given, assume
from hypothesis.strategies import integers, composite

from s3torchconnector.s3reader.dcp_optimized import (
    DCPOptimizedS3Reader,
    ItemRange,
    RangeGroup,
    _ItemViewBuffer,
    DEFAULT_MAX_GAP_SIZE,
    FIND_ITEM_ERROR_PREFIX,
)
from .test_s3reader_common import (
    TEST_BUCKET,
    TEST_KEY,
    MOCK_OBJECT_INFO,
    MOCK_STREAM,
    create_object_info_getter,
    create_stream_getter,
    bytestream_and_positions,
)


def create_dcp_s3reader(
    ranges: Optional[List[ItemRange]] = None,
    stream_data: Optional[List[bytes]] = None,
    max_gap_size: int = DEFAULT_MAX_GAP_SIZE,
    chunk_size: int = 5,
):
    """Create DCPOptimizedS3Reader with mock stream data"""
    if ranges is None:
        ranges = [ItemRange(0, 10)]
    if stream_data is None:
        stream_data = [b"0123456789"]

    return DCPOptimizedS3Reader(
        TEST_BUCKET,
        TEST_KEY,
        ranges,
        create_object_info_getter(stream_data),
        create_stream_getter(stream_data, chunk_size=chunk_size),  # type: ignore
        max_gap_size=max_gap_size,
    )


@composite
def dcp_ranges_and_stream(draw):
    """Generate sorted and non-overlapping item ranges with corresponding stream data"""
    num_ranges = draw(integers(min_value=10, max_value=100))
    ranges = []
    current_pos = 0

    for _ in range(num_ranges):
        # Random gap size and random length
        start = current_pos + draw(integers(min_value=0, max_value=2000))
        length = draw(integers(min_value=100, max_value=1000))
        end = start + length
        ranges.append(ItemRange(start, end))
        current_pos = end

    stream_data = [b"x" * current_pos]

    return ranges, stream_data


class TestItemViewBuffer:
    """ItemViewBuffer Tests"""

    def test_append_view_and_size_calculation(self):
        """Test append_view and size calculation correctness"""
        buffer = _ItemViewBuffer()
        assert buffer._size == 0
        assert buffer.tell() == 0

        buffer.append_view(memoryview(b"Hello"))
        assert buffer._size == 5
        assert len(buffer._segments) == 1
        assert buffer._segments[0] == b"Hello"
        assert buffer._lengths[0] == 5
        assert buffer._offsets[0] == 0

        buffer.append_view(memoryview(b" World!"))
        assert buffer._size == 12
        assert len(buffer._segments) == 2
        assert buffer._segments[1] == b" World!"
        assert buffer._lengths[1] == 7
        assert buffer._offsets[1] == 5

        # Empty views should be ignored
        buffer.append_view(memoryview(b""))
        assert buffer._size == 12
        assert len(buffer._segments) == 2

    @pytest.mark.parametrize(
        "start_pos, offset, whence, expected_pos",
        [
            # SEEK_SET
            (0, 5, SEEK_SET, 5),
            (3, 0, SEEK_SET, 0),
            (5, 15, SEEK_SET, 15),  # Allow seeking past buffer end
            # SEEK_CUR
            (3, 2, SEEK_CUR, 5),
            (5, -2, SEEK_CUR, 3),
            (5, 10, SEEK_CUR, 15),
            # SEEK_END
            (0, -3, SEEK_END, 7),
            (5, 0, SEEK_END, 10),
            (5, 2, SEEK_END, 12),
        ],
    )
    def test_buffer_seek_all_modes(self, start_pos, offset, whence, expected_pos):
        buffer = _ItemViewBuffer()
        buffer.append_view(memoryview(b"0123456789"))
        buffer.seek(start_pos)

        pos = buffer.seek(offset, whence)
        assert pos == expected_pos
        assert buffer.tell() == expected_pos

    @pytest.mark.parametrize(
        "offset, whence, expected_error",
        [
            # Negative offsets
            (-1, SEEK_SET, AssertionError),
            (-1, SEEK_CUR, AssertionError),
            (-15, SEEK_END, AssertionError),
            # Invalid whence
            (5, 3, ValueError),
            (5, None, ValueError),
            (5, "SEEK_SET", ValueError),
        ],
    )
    def test_buffer_invalid_seek(self, offset, whence, expected_error):
        buffer = _ItemViewBuffer()
        buffer.append_view(memoryview(b"0123456789"))

        with pytest.raises(expected_error):
            buffer.seek(offset, whence)

    @pytest.mark.parametrize(
        "start, size, expected_data, expected_pos",
        [
            # Zero reads
            (0, 0, b"", 0),
            (5, 0, b"", 5),
            (10, 0, b"", 10),
            # Normal reads
            (0, 4, b"0123", 4),
            (2, 3, b"234", 5),
            (5, 5, b"56789", 10),
            (0, 10, b"0123456789", 10),
            # Past EOF reads
            (8, 5, b"89", 10),
            (9, 3, b"9", 10),
            (0, 15, b"0123456789", 10),
            # EOF and beyond EOF reads
            (10, 5, b"", 10),
            (15, 5, b"", 15),
        ],
    )
    def test_buffer_read_cases(self, start, size, expected_data, expected_pos):
        """Test read() normal reads and edge cases"""
        buffer = _ItemViewBuffer()
        for segment in [b"012", b"34", b"5", b"6789"]:
            buffer.append_view(memoryview(segment))

        buffer.seek(start)
        data = buffer.read(size)
        assert data == expected_data
        assert buffer.tell() == expected_pos

    @pytest.mark.parametrize(
        "segments",
        [
            # Fast path works (first segment >= 4 bytes)
            [b"PK\x03\x04abcdef"],
            # Fast path does not trigger, but still reads correctly
            [b"PK", b"\x03\x04abcdef"],
            [b"P", b"K\x03", b"\x04abcdef"],
            [b"P", b"K", b"\x03", b"\x04abcdef"],
        ],
    )
    def test_buffer_read_fast_path_optimization(self, segments):
        """Test read(4) at pos=0 fast path"""
        buffer = _ItemViewBuffer()
        for segment in segments:
            buffer.append_view(memoryview(segment))

        # Test read(4) at pos=0
        data = buffer.read(4)
        assert data == b"PK\x03\x04"
        assert buffer.tell() == 4
        assert isinstance(data, bytes)

        # Test normal read continues
        data = buffer.read(3)
        assert data == b"abc"
        assert buffer.tell() == 7

    @pytest.mark.parametrize(
        "buf",
        [
            bytearray(5),
            memoryview(bytearray(5)),
        ],
    )
    def test_buffer_readinto_valid_types(self, buf):
        """Test readinto() valid buffer types"""
        buffer = _ItemViewBuffer()
        buffer.append_view(memoryview(b"hello"))

        bytes_read = buffer.readinto(buf)

        assert bytes_read == 5
        assert bytes(buf) == b"hello"
        assert buffer.tell() == 5

    @pytest.mark.parametrize(
        "buf, expected_error",
        [
            # Invalid
            ("hello", TypeError),
            (12345, TypeError),
            ([1, 2, 3, 4, 5], TypeError),
            (None, TypeError),
            # Readonly memoryviews
            (memoryview(b"12345"), AssertionError),
            (memoryview(bytearray(5)).toreadonly(), AssertionError),
        ],
    )
    def test_buffer_readinto_invalid_types(self, buf, expected_error):
        """Test readinto() invalid buffer types"""
        buffer = _ItemViewBuffer()
        buffer.append_view(memoryview(b"hello"))

        with pytest.raises(expected_error):
            buffer.readinto(buf)

    @pytest.mark.parametrize(
        "start, buf_size, expected_bytes, expected_pos",
        [
            # Zero buffer cases
            (0, 0, 0, 0),
            (5, 0, 0, 5),
            (10, 0, 0, 10),
            # Normal readinto cases
            (0, 5, 5, 5),
            (3, 4, 4, 7),
            (7, 2, 2, 9),
            # Near EOF cases
            (8, 5, 2, 10),
            (9, 3, 1, 10),
            (9, 1, 1, 10),
            # EOF and beyond EOF cases
            (10, 5, 0, 10),
            (15, 5, 0, 15),
        ],
    )
    def test_buffer_readinto_edge_cases(
        self, start, buf_size, expected_bytes, expected_pos
    ):
        """Test readinto() normal read and edge cases"""
        buffer = _ItemViewBuffer()
        for segment in [b"012", b"34", b"5", b"6789"]:
            buffer.append_view(memoryview(segment))

        buffer.seek(start)
        buf = bytearray(buf_size)
        bytes_read = buffer.readinto(buf)

        assert bytes_read == expected_bytes
        assert buffer.tell() == expected_pos

        if expected_bytes > 0:
            expected_data = b"0123456789"[start : start + expected_bytes]
            assert buf[:expected_bytes] == expected_data

    @pytest.mark.parametrize("buf_size", [1, 4, 10, 50])
    @pytest.mark.parametrize(
        "buf_type", [bytearray, lambda size: memoryview(bytearray(size))]
    )
    @given(bytestream_and_positions())
    def test_buffer_readinto_hypothesis(
        self, buf_size, buf_type, stream_and_positions: Tuple[List[bytes], List[int]]
    ):
        """Test readinto() operations against BytesIO equivalent"""
        segments, read_positions = stream_and_positions

        buffer = _ItemViewBuffer()
        for segment in segments:
            buffer.append_view(memoryview(segment))
        reference_io = BytesIO(b"".join(segments))

        for pos in read_positions:
            buffer.seek(pos)
            reference_io.seek(pos)

            buf = buf_type(buf_size)
            ref_buf = buf_type(buf_size)

            bytes_read = buffer.readinto(buf)
            ref_bytes_read = reference_io.readinto(ref_buf)

            assert bytes_read == ref_bytes_read
            assert buffer.tell() == reference_io.tell()
            assert bytes(buf[:bytes_read]) == bytes(ref_buf[:bytes_read])


class TestCreationAndValidation:
    """
    DCPOptimizedS3Reader creation and validation tests.
    References some tests in test_s3reader_common.py; not tested there since its behaviour is different.
    """

    def test_s3reader_creation(self):
        """Test basic reader creation"""
        reader = create_dcp_s3reader()
        assert reader
        assert reader.bucket == TEST_BUCKET
        assert reader.key == TEST_KEY
        assert not reader.closed

    @pytest.mark.parametrize(
        "bucket, key, expected_error",
        [
            (None, TEST_KEY, "Bucket should be specified"),
            ("", TEST_KEY, "Bucket should be specified"),
            (TEST_BUCKET, None, "Key should be specified"),
            (TEST_BUCKET, "", "Key should be specified"),
        ],
    )
    def test_invalid_bucket_key_validation(self, bucket, key, expected_error):
        """Test bucket and key validation"""
        ranges = [ItemRange(0, 10)]
        with pytest.raises(ValueError, match=expected_error):
            DCPOptimizedS3Reader(bucket, key, ranges, MOCK_OBJECT_INFO, MOCK_STREAM)

    @pytest.mark.parametrize(
        "offset",
        [0.4, 0.0, 1.0, "test", 1 + 2j, [1, 2, 3], {}, {2}],
    )
    def test_fails_with_non_int_arg(self, offset):
        """Test type validation for seek and read arguments"""
        reader = create_dcp_s3reader()

        with pytest.raises(TypeError):
            reader.seek(offset)
        with pytest.raises(TypeError):
            reader.read(offset)

    def test_empty_ranges_rejection(self):
        """Test empty ranges are rejected"""
        with pytest.raises(
            ValueError,
            match=r"item_ranges must be a non-empty List\[ItemRange\] object",
        ):
            create_dcp_s3reader(ranges=[])

    @pytest.mark.parametrize("max_gap_size", [0, 1024, 32 * 1024 * 1024])
    def test_valid_max_gap_size(self, max_gap_size):
        """Test valid max_gap_size values are accepted"""
        reader = create_dcp_s3reader(max_gap_size=max_gap_size)
        assert reader._max_gap_size == max_gap_size

    @pytest.mark.parametrize(
        "max_gap_size,expected_error,error_msg",
        [
            (-1, ValueError, "max_gap_size must be non-negative"),
            ("1", TypeError, "max_gap_size must be int or float, got str"),
            ([1], TypeError, "max_gap_size must be int or float, got list"),
            (None, TypeError, "max_gap_size must be int or float, got NoneType"),
        ],
    )
    def test_invalid_max_gap_size_types(self, max_gap_size, expected_error, error_msg):
        """Test max_gap_size type validation"""
        with pytest.raises(expected_error, match=error_msg):
            create_dcp_s3reader(max_gap_size=max_gap_size)


class TestValidateAndCoalesceRanges:
    """DCPOptimizedS3Reader _validate_and_coalesce_ranges tests for different ItemRanges and max_gap_sizes"""

    @pytest.mark.parametrize(
        "ranges,error_msg",
        [
            ([ItemRange(-1, 10)], "Invalid range: -1-10"),
            ([ItemRange(0, 5), ItemRange(10, 5)], "Invalid range: 10-5"),
            ([ItemRange(10, 20), ItemRange(5, 10)], "Unsorted ranges: 10-20 and 5-10"),
            ([ItemRange(0, 10), ItemRange(5, 15)], "Overlapping ranges: 0-10 and 5-15"),
        ],
    )
    def test_validation_errors(self, ranges, error_msg):
        """Test validation error cases"""
        with pytest.raises(ValueError, match=error_msg):
            create_dcp_s3reader(ranges)

    def test_empty_ranges_filtered_out(self):
        """Test empty ranges are filtered out (during initialization)"""
        ranges = [ItemRange(10, 10), ItemRange(20, 30), ItemRange(100, 100)]
        reader = create_dcp_s3reader(ranges)

        assert len(reader._item_ranges) == 1
        assert reader._item_ranges[0].start == 20
        assert reader._item_ranges[0].end == 30

    def test_all_empty_ranges_error(self):
        """Test all empty ranges causes error"""
        ranges = [ItemRange(10, 10), ItemRange(20, 20)]

        with pytest.raises(ValueError, match="No non-empty ranges to read"):
            create_dcp_s3reader(ranges)

    @pytest.mark.parametrize(
        "max_gap_size,ranges,expected_groups",
        [
            # Basic Tests
            (10, [ItemRange(0, 10)], 1),  # Single range
            (0, [ItemRange(0, 10), ItemRange(20, 30)], 2),  # No coalescing
            (10, [ItemRange(0, 10), ItemRange(20, 30)], 1),  # Just coalesced
            (9, [ItemRange(0, 10), ItemRange(20, 30)], 2),  # Just no coalesce
            # 3 ranges
            (50, [ItemRange(0, 10), ItemRange(20, 30), ItemRange(100, 110)], 2),
            (50, [ItemRange(0, 50), ItemRange(50, 100), ItemRange(149, 199)], 1),
            # Zero gap
            (0, [ItemRange(0, 10), ItemRange(10, 20)], 1),
            (0, [ItemRange(0, 10), ItemRange(11, 20)], 2),
            # Infinite / large gap size - coalesce all
            (float("inf"), [ItemRange(0, 10), ItemRange(1016 * 1024, 1024 * 1024)], 1),
            (sys.maxsize, [ItemRange(0, 10), ItemRange(1016 * 1024, 1024 * 1024)], 1),
            (2**50, [ItemRange(0, 10), ItemRange(1016 * 1024, 1024 * 1024)], 1),
        ],
    )
    def test_coalescing_behaviour(self, max_gap_size, ranges, expected_groups):
        """Test coalescing with different max_gap_sizes and edge cases"""
        stream_data = [b"x" * 200]  # Enough data for all ranges
        reader = create_dcp_s3reader(ranges, stream_data, max_gap_size)
        assert len(reader._range_groups) == expected_groups

    @given(dcp_ranges_and_stream(), integers(min_value=5, max_value=25))
    def test_coalescing_behaviour_hypothesis(self, ranges_and_stream, max_gap_size):
        """Check coalescing correctness for different inputs"""
        ranges, stream_data = ranges_and_stream
        assume(len(ranges) > 1)

        reader = create_dcp_s3reader(ranges, stream_data, max_gap_size=max_gap_size)
        groups = reader._range_groups

        # All ranges in all groups are covered (and are in the same order)
        covered_ranges = [r for group in groups for r in group.item_ranges]
        assert covered_ranges == ranges

        # Groups separated by more than max_gap_size
        for i in range(1, len(groups)):
            gap = groups[i].start - groups[i - 1].end
            assert gap > max_gap_size

        # ItemRanges within groups less than or equal to max_gap_size
        for group in groups:
            for i in range(1, len(group.item_ranges)):
                gap = group.item_ranges[i].start - group.item_ranges[i - 1].end
                assert gap <= max_gap_size

    def test_group_start_to_group_mapping(self):
        """Test _group_start_to_group correctness, generated after _validate_and_coalesce_ranges method"""
        ranges = [ItemRange(0, 10), ItemRange(50, 60), ItemRange(70, 80)]
        reader = create_dcp_s3reader(ranges, [b"x" * 100], max_gap_size=15)

        # Should create 2 groups: [0-10] and [50-80]
        assert len(reader._range_groups) == 2

        # Check group starts
        expected_group_starts = [0, 50]
        actual_group_starts = [group.start for group in reader._range_groups]
        assert actual_group_starts == expected_group_starts

        # Check group start mappings
        assert set(reader._group_start_to_group.keys()) == {0, 50}
        assert reader._group_start_to_group[0].start == 0
        assert reader._group_start_to_group[50].start == 50

        # Check 70 is not group start (part of 2nd group, not a group start)
        assert 70 not in reader._group_start_to_group


class TestStreamManagement:
    """Tests for _get_stream_for_item and how stream and left data is managed within and between range groups"""

    def test_coalesced_ranges_stream_reuse(self):
        """Test stream reuse within coalesced group vs separate streams for different groups"""
        # 3 ranges: first 2 coalesce (gap=5 ≤ 10), third is separate (gap=85 > 10)
        ranges = [ItemRange(0, 10), ItemRange(15, 25), ItemRange(110, 120)]
        test_data = [b"0123456789-----abcdefghij" + b"x" * 85 + b"ABCDEFGHIJ"]

        stream_calls = []

        def spy_get_stream(start=None, end=None):
            stream_calls.append((start, end))
            return create_stream_getter(test_data)(start, end)

        reader = DCPOptimizedS3Reader(
            TEST_BUCKET,
            TEST_KEY,
            ranges,
            create_object_info_getter(test_data),
            spy_get_stream,  # type: ignore
            max_gap_size=10,
        )

        # 2 groups: [0-25] and [110-120]
        assert len(reader._range_groups) == 2
        assert reader._range_groups[0].start == 0
        assert reader._range_groups[0].end == 25
        assert reader._range_groups[1].start == 110
        assert reader._range_groups[1].end == 120

        # Read from first group
        reader.seek(0)
        assert reader.read(10) == b"0123456789"
        reader.seek(15)
        assert reader.read(10) == b"abcdefghij"
        # Only 1 stream call so far for both ItemRanges
        assert stream_calls == [(0, 25)]

        # Read from second group
        reader.seek(110)
        assert reader.read(10) == b"ABCDEFGHIJ"
        # 2 stream calls
        assert stream_calls == [(0, 25), (110, 120)]

    def test_leftover_handling_with_chunks(self):
        """Test leftover data handling across items with chunk boundaries"""
        ranges = [ItemRange(0, 7), ItemRange(12, 18), ItemRange(21, 25)]
        test_data = [b"0123456789ABCDEFGHIJabcdefghij"]
        # Chunk size 5 - each next() iteration will return 5 bytes of data
        reader = create_dcp_s3reader(ranges, test_data, max_gap_size=10, chunk_size=5)

        # Should coalesce into single group
        assert len(reader._range_groups) == 1

        # Read first item
        reader.seek(0)
        assert reader.read(7) == b"0123456"
        assert reader._stream_state.leftover
        assert bytes(reader._stream_state.leftover) == b"789"  # bytes 8-10

        # Read second item (should use leftover data)
        reader.seek(12)
        assert reader.read(6) == b"CDEFGH"
        assert reader._stream_state.leftover
        assert bytes(reader._stream_state.leftover) == b"IJ"  # bytes 19-20

    def test_get_stream_for_item_missing_stream_error(self):
        """Test _get_stream_for_item error when stream is None for non-first item"""
        ranges = [ItemRange(0, 10), ItemRange(15, 25)]
        reader = create_dcp_s3reader(ranges, [b"x" * 30], max_gap_size=10)

        # Corrupt state: advance to second item without creating stream
        reader._current_item = ranges[1]
        reader._stream_state.stream = None  # force error (should not happen normally)

        with pytest.raises(AssertionError):
            reader._get_stream_for_item(ranges[1])


class TestReaderIO:
    """Reader Interface (seek/read/readinto) and Sequential Access Tests"""

    @pytest.mark.parametrize(
        "offset, whence, expected_error, error_msg",
        [
            ("5", SEEK_SET, TypeError, "integer argument expected, got <class 'str'>"),
            (0, SEEK_END, ValueError, "whence must be SEEK_CUR or SEEK_SET integers"),
            (-1, SEEK_SET, ValueError, "negative seek value -1"),
        ],
    )
    def test_seek_invalid_types(self, offset, whence, expected_error, error_msg):
        """Test seek() parameter validation"""
        reader = create_dcp_s3reader()
        with pytest.raises(expected_error, match=error_msg):
            reader.seek(offset, whence)

    @pytest.mark.parametrize(
        "size, expected_error, error_msg",
        [
            (None, ValueError, "Size cannot be None; full read not supported"),
            (-1, ValueError, "Size cannot be negative; full read not supported"),
            ("5", TypeError, "argument should be integer or None, not <class 'str'>"),
        ],
    )
    def test_read_invalid_types(self, size, expected_error, error_msg):
        """Test read() parameter validation"""
        reader = create_dcp_s3reader()
        with pytest.raises(expected_error, match=error_msg):
            reader.read(size)

    def test_read_zero_size(self):
        """Test read(0) returns empty bytes"""
        reader = create_dcp_s3reader()
        assert reader.read(0) == b""

    @pytest.mark.parametrize(
        "buf, expected_error",
        [
            ("hello", TypeError),
            (12345, TypeError),
            ([1, 2, 3], TypeError),
            (None, TypeError),
            (memoryview(b"test"), AssertionError),  # _ItemViewBuffer check
        ],
    )
    def test_readinto_invalid_types(self, buf, expected_error):
        """Test readinto() parameter validation"""
        reader = create_dcp_s3reader()
        with pytest.raises(expected_error):
            reader.readinto(buf)

    @pytest.mark.parametrize("buf", [bytearray(5), memoryview(bytearray(5))])
    def test_readinto_valid_types(self, buf):
        """Test readinto() accepts valid buffer types"""
        reader = create_dcp_s3reader()

        bytes_read = reader.readinto(buf)
        assert bytes_read == 5
        assert bytes(buf) == b"01234"

    def test_sequential_access_enforcement(self):
        """Test sequential access pattern enforcement"""
        ranges = [ItemRange(0, 10), ItemRange(20, 30)]
        stream_data = [b"0123456789" + b"x" * 10 + b"abcdefghij"]
        reader = create_dcp_s3reader(ranges, stream_data)

        # Forward access should work
        reader.seek(0)
        assert reader.read(5) == b"01234"
        assert reader.read(5) == b"56789"

        # Move to next item
        reader.seek(20)
        assert reader.read(5) == b"abcde"

        # Backward access to previous item should fail
        reader.seek(5)
        with pytest.raises(ValueError, match="Position 5 before current range 20-30"):
            reader.read(1)

    def test_within_item_seeking(self):
        """Test seeking within current item is allowed"""
        ranges = [ItemRange(0, 20)]
        reader = create_dcp_s3reader(ranges, [b"0123456789abcdefghij"])

        reader.seek(5)
        assert reader.read(5) == b"56789"

        # Seek backward within same item
        reader.seek(2)
        assert reader.read(3) == b"234"

    @pytest.mark.parametrize(
        "pos, setup_reads, error_suffix",
        [
            (5, [], "Position 5 before current range 10-20"),
            (25, [10], "Position 25 in gap between ranges 10-20 and 30-40"),
            (45, [10], "Position 45 beyond next range 30-40"),
            (45, [10, 30], "Position 45 beyond last range 30-40"),
            (15, [10, 30], "Position 15 before current range 30-40"),
        ],
    )
    def test_find_item_for_position_errors(self, pos, setup_reads, error_suffix):
        """Test seeking to position outside valid ranges for _find_item_for_position"""
        ranges = [ItemRange(10, 20), ItemRange(30, 40)]
        reader = create_dcp_s3reader(ranges, [b"x" * 50])

        # Setup reads to advance iterator state
        for read_pos in setup_reads:
            reader.seek(read_pos)
            reader.read(1)

        reader.seek(pos)
        expected_pattern = re.escape(FIND_ITEM_ERROR_PREFIX) + re.escape(error_suffix)
        with pytest.raises(ValueError, match=expected_pattern):
            reader.read(1)

    @pytest.mark.parametrize(
        "ranges, read_pattern, expected_data",
        [
            # Single range
            ([ItemRange(0, 5)], [(0, 5)], [b"01234"]),
            # 2 ranges
            (
                [ItemRange(0, 5), ItemRange(10, 15)],
                [(0, 3), (3, 2), (10, 5)],
                [b"012", b"34", b"abcde"],
            ),
            # Should stop at item boundary
            (
                [ItemRange(0, 5), ItemRange(10, 15)],
                [(3, 5)],
                [b"34"],
            ),
        ],
    )
    def test_read_patterns(self, ranges, read_pattern, expected_data):
        """Test various read patterns and item boundary behaviour"""
        test_data = [b"0123456789abcdefghij"]
        reader = create_dcp_s3reader(ranges, test_data)

        results = []
        for pos, size in read_pattern:
            reader.seek(pos)
            data = reader.read(size)
            results.append(data)

        assert results == expected_data

    @given(dcp_ranges_and_stream())
    def test_sequential_io_hypothesis(self, ranges_and_stream):
        """Quick integration test to read all ItemRanges sequentially"""
        ranges, stream_data = ranges_and_stream
        assume(len(ranges) > 0)

        reader = create_dcp_s3reader(ranges, stream_data)

        for range_item in ranges:
            reader.seek(range_item.start)
            range_size = range_item.end - range_item.start
            data = reader.read(range_size)

            assert len(data) == range_size
            assert reader.tell() == range_item.end

    def test_close(self):
        """Test close() behaviour"""
        reader = create_dcp_s3reader()
        reader.close()

        assert reader.closed
        assert reader._stream_state.stream is None
        assert reader._stream_state.leftover is None
        assert reader._current_item_buffer is None
