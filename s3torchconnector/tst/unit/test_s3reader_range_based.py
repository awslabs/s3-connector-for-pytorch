#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD


import logging
import sys
from io import BytesIO, SEEK_END, SEEK_CUR
from typing import List, Tuple
from unittest.mock import Mock

import pytest
from hypothesis import given, assume
from hypothesis.strategies import lists, binary, integers, composite

from s3torchconnector.s3reader import RangedS3Reader
from s3torchconnector.s3reader.ranged import DEFAULT_BUFFER_SIZE
from .test_s3reader_common import (
    TEST_BUCKET,
    TEST_KEY,
    MOCK_OBJECT_INFO,
    MOCK_STREAM,
    bytestream_and_positions,
    bytestream_and_position,
    create_object_info_getter,
    create_stream_getter,
)

logging.basicConfig(
    format="%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s"
)
logging.getLogger().setLevel(1)

log = logging.getLogger(__name__)

BUFFER_BEHAVIOR_TEST_CASES = [
    # buffer_size, read_size, enable_buffering, buffer_should_load
    (256, 50, True, True),  # Small read - should load buffer
    (256, 255, True, True),  # Just under buffer_size - should load buffer
    (256, 256, True, False),  # Equal to buffer_size - should bypass buffer
    (256, 257, True, False),  # Just over buffer_size - should bypass buffer
    (256, 300, True, False),  # Large read - should bypass buffer
    (0, 50, False, False),  # Buffering disabled - should bypass buffer
    (256, 0, True, False),  # Zero read - should not load buffer (early return)
]


def create_range_s3reader(stream, buffer_size=None):
    return RangedS3Reader(
        TEST_BUCKET,
        TEST_KEY,
        create_object_info_getter(stream),
        create_stream_getter(stream),
        buffer_size=buffer_size,
    )


# These initial tests were separated from test_s3reader_common due to difference in operation to sequential reader
@given(lists(binary(min_size=1, max_size=5000)))
def test_s3reader_writes_size_before_read_all(stream):
    s3reader = create_range_s3reader(stream)
    assert s3reader._size is None
    total_length = sum(map(len, stream))
    # We're able to read all the data
    assert len(s3reader.read(total_length)) == total_length
    # Read operation writes size before reading
    assert s3reader._size == total_length
    # Reading past the end gives us empty
    assert s3reader.read(1) == b""


@given(
    lists(binary(min_size=20, max_size=30), min_size=0, max_size=2),
    integers(min_value=0, max_value=10),
)
def test_s3reader_writes_size_when_readinto_buffer_smaller_than_chunks(
    stream, buf_size
):
    s3reader = create_range_s3reader(stream)
    assert s3reader._size is None
    total_length = sum(map(len, stream))
    buf = memoryview(bytearray(buf_size))
    # We're able to read all the available data or the data that can be accommodated in buf
    if buf_size > 0 and total_length > 0:
        assert s3reader.readinto(buf) == buf_size
        assert s3reader.tell() == buf_size
        # Readinto operation does write size
        assert s3reader._size == total_length
        # confirm that read data is the same as in source
        assert buf[:buf_size] == b"".join(stream)[:buf_size]
    else:
        assert s3reader.readinto(buf) == 0
        assert s3reader.tell() == 0


@pytest.mark.parametrize(
    "invalid_buf",
    [
        b"test",  # bytes (readonly)
        memoryview(b"test"),  # memoryview of bytes (readonly)
        42,  # non-buffer type
        "string",  # non-buffer type
        None,  # non-buffer type
    ],
    ids=["bytes", "readonly_memoryview", "int", "str", "none"],
)
def test_s3reader_readinto_invalid_buffer(invalid_buf):
    """Test that readinto raises TypeError for readonly and non-buffer objects"""
    s3reader = create_range_s3reader([b"test data"])
    with pytest.raises(
        TypeError, match="argument must be a writable bytes-like object"
    ):
        s3reader.readinto(invalid_buf)


# Buffer Behaviour Tests


@pytest.mark.parametrize(
    "buffer_size, expected_buffer_size, expected_enable_buffering",
    [
        (None, DEFAULT_BUFFER_SIZE, True),
        (1024, 1024, True),
        (1024 * 1024, 1024 * 1024, True),
        (64 * 1024 * 1024, 64 * 1024 * 1024, True),
        (0, 0, False),
    ],
)
def test_buffer_configuration(
    buffer_size,
    expected_buffer_size,
    expected_enable_buffering,
):
    """Test buffer configuration with different sizes"""
    s3reader = create_range_s3reader(MOCK_STREAM, buffer_size)

    assert s3reader._buffer_size == expected_buffer_size
    assert s3reader._enable_buffering is expected_enable_buffering


def _verify_buffer_configuration(s3reader, buffer_size, enable_buffering):
    """Helper to verify buffer configuration"""
    assert s3reader._enable_buffering == enable_buffering
    if enable_buffering:
        assert isinstance(s3reader._buffer, bytearray)
        assert s3reader._buffer_size == buffer_size
    else:
        assert s3reader._buffer is None
        assert s3reader._buffer_size == 0


def _verify_buffer_load_behavior(s3reader, buffer_state, buffer_should_load):
    """Helper to verify buffer loading behavior and content"""
    total_data, buffer_size, initial_buffer_end, data = buffer_state

    if buffer_should_load:
        # Verify _buffer_end is tracked correctly
        assert s3reader._buffer_end > initial_buffer_end  # Buffer was loaded
        expected_buffer_end = min(s3reader._buffer_start + buffer_size, len(total_data))
        assert s3reader._buffer_end == expected_buffer_end
        # Verify buffer contains correct data
        expected_buffer_data = total_data[:expected_buffer_end]
        actual_buffer_data = s3reader._buffer[: len(expected_buffer_data)]
        assert actual_buffer_data == expected_buffer_data
        assert s3reader._buffer[: len(data)] == data
    else:
        assert s3reader._buffer_end == initial_buffer_end  # Buffer unchanged


@pytest.mark.parametrize(
    "buffer_size, read_size, enable_buffering, buffer_should_load",
    BUFFER_BEHAVIOR_TEST_CASES,
)
@given(lists(binary(min_size=100, max_size=200), min_size=10, max_size=20))
def test_buffer_behavior_by_read_size(
    buffer_size, read_size, enable_buffering, buffer_should_load, stream
):
    """Test buffer behavior with different read sizes including edge cases"""
    total_data = b"".join(stream)
    s3reader = create_range_s3reader(stream, buffer_size)

    _verify_buffer_configuration(s3reader, buffer_size, enable_buffering)

    initial_buffer_end = s3reader._buffer_end
    data = s3reader.read(read_size)

    buffer_state = (total_data, buffer_size, initial_buffer_end, data)
    _verify_buffer_load_behavior(s3reader, buffer_state, buffer_should_load)

    assert len(data) == read_size
    assert data == total_data[:read_size]


@pytest.mark.parametrize(
    "buffer_size, buf_size, enable_buffering, buffer_should_load",
    BUFFER_BEHAVIOR_TEST_CASES,
)
@given(lists(binary(min_size=100, max_size=200), min_size=10, max_size=20))
def test_buffer_behavior_by_readinto_size(
    buffer_size, buf_size, enable_buffering, buffer_should_load, stream
):
    """Test buffer behavior with readinto using different buffer sizes including edge cases"""
    total_data = b"".join(stream)
    s3reader = create_range_s3reader(stream, buffer_size)

    _verify_buffer_configuration(s3reader, buffer_size, enable_buffering)

    initial_buffer_end = s3reader._buffer_end
    buf = bytearray(buf_size)
    bytes_read = s3reader.readinto(buf)

    buffer_state = (total_data, buffer_size, initial_buffer_end, buf)
    _verify_buffer_load_behavior(s3reader, buffer_state, buffer_should_load)

    assert bytes_read == buf_size
    assert buf == total_data[:buf_size]


@given(lists(binary(min_size=50, max_size=100), min_size=5, max_size=10))
def test_buffer_reuse_sequential_reads(stream):
    """Test that sequential small reads reuse the buffer efficiently"""
    buffer_size = 200
    total_data = b"".join(stream)

    s3reader = create_range_s3reader(stream, buffer_size)

    # First read loads buffer
    first_read = s3reader.read(20)
    initial_buffer_start = s3reader._buffer_start
    initial_buffer_end = s3reader._buffer_end

    # Verify buffer was loaded
    assert s3reader.tell() == 20
    assert initial_buffer_end == min(buffer_size, len(total_data))
    expected_buffer_data = total_data[:initial_buffer_end]
    assert s3reader._buffer[: len(expected_buffer_data)] == expected_buffer_data

    # Second read within buffer range
    second_read = s3reader.read(20)
    assert s3reader.tell() == 40
    assert s3reader._buffer_start == initial_buffer_start  # Buffer not reloaded
    assert s3reader._buffer_end == initial_buffer_end
    assert first_read + second_read == total_data[:40]

    # Seek within buffer range
    s3reader.seek(20, SEEK_CUR)
    assert s3reader.tell() == 60
    # Third read within buffer range
    third_read = s3reader.read(20)
    assert s3reader.tell() == 80
    assert s3reader._buffer_start == initial_buffer_start  # Buffer not reloaded
    assert s3reader._buffer_end == initial_buffer_end
    assert third_read == total_data[60:80]


@given(lists(binary(min_size=100, max_size=150), min_size=8, max_size=12))
def test_buffer_reload_on_read_outside_range(stream):
    """Test buffer reload when reading outside current buffer range"""
    buffer_size = 200
    total_data = b"".join(stream)

    s3reader = create_range_s3reader(stream, buffer_size)

    # Load initial buffer at position 0
    s3reader.read(10)
    initial_buffer_start = s3reader._buffer_start
    initial_buffer_end = s3reader._buffer_end
    assert initial_buffer_start == 0
    assert initial_buffer_end == min(buffer_size, len(total_data))

    # Seek to position outside buffer range
    outside_position = buffer_size + 20
    s3reader.seek(outside_position)
    data = s3reader.read(10)

    # Buffer should be reloaded at new position
    assert s3reader._buffer_start != initial_buffer_start
    assert s3reader._buffer_end != initial_buffer_end
    assert s3reader._buffer_start == outside_position
    assert s3reader._buffer_end == min(outside_position + buffer_size, len(total_data))
    assert data == total_data[outside_position : outside_position + 10]


@pytest.mark.parametrize(
    "total_size, buffer_size, read_position, read_size, expected_data_len, expected_data",
    [
        # buffer_size > stream size, read near end (20 bytes left)
        (100, 150, 80, 30, 20, b"A" * 20),
        # buffer_size > stream size, read at end (5 bytes left)
        (100, 150, 95, 10, 5, b"A" * 5),
        # buffer_size < stream size, read past end (10 bytes left)
        (100, 80, 90, 15, 10, b"A" * 10),
        # Read exactly at object end (no bytes left)
        (100, 80, 100, 5, 0, b""),
    ],
)
def test_buffer_at_object_end_boundary(
    total_size,
    buffer_size,
    read_position,
    read_size,
    expected_data_len,
    expected_data,
):
    """Test buffer behavior when reading near/at object end"""
    stream = [b"A" * total_size]
    s3reader = create_range_s3reader(stream, buffer_size)

    s3reader.seek(read_position)
    data = s3reader.read(read_size)

    # Verify buffer boundaries respect object size
    if s3reader._enable_buffering and s3reader._buffer_end > 0:
        assert s3reader._buffer_end <= total_size  # Never exceed object size
        assert s3reader._buffer_start <= read_position

    # Verify exact expected results
    assert len(data) == expected_data_len
    assert data == expected_data
