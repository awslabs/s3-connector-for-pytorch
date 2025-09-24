#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import logging
import sys
from io import BytesIO, SEEK_SET, SEEK_END, SEEK_CUR
from typing import List, Tuple, Type
from unittest.mock import Mock

import pytest
from hypothesis import given, assume
from hypothesis.strategies import lists, binary, integers, composite
from s3torchconnectorclient._mountpoint_s3_client import ObjectInfo, GetObjectStream

from s3torchconnector import S3Reader
from s3torchconnector.s3reader import SequentialS3Reader, RangedS3Reader

logging.basicConfig(
    format="%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s"
)
logging.getLogger().setLevel(1)

log = logging.getLogger(__name__)

TEST_BUCKET = "test-bucket"
TEST_KEY = "test-key"
MOCK_OBJECT_INFO = Mock(ObjectInfo)
MOCK_STREAM = Mock(GetObjectStream)


@pytest.fixture(
    params=[
        SequentialS3Reader,
        lambda *args, **kwargs: RangedS3Reader(*args, **kwargs),  # Default buffer
        lambda *args, **kwargs: RangedS3Reader(
            *args, **kwargs, buffer_size=0
        ),  # No buffer
    ],
    ids=["sequential", "range_based_with_buffer", "range_based_no_buffer"],
    scope="module",
)
def reader_implementation(request) -> Type[S3Reader]:
    """Provide S3Reader implementations for all supported reader types and buffer configurations."""
    return request.param


def create_s3reader(stream, reader_implementation: Type[S3Reader]) -> S3Reader:
    return reader_implementation(
        TEST_BUCKET,
        TEST_KEY,
        create_object_info_getter(stream),
        create_stream_getter(stream),
    )


def create_object_info_getter(stream_data):
    """Create an object info getter function with size calculated from stream data."""

    def get_object_info():
        mock_object_info = Mock(ObjectInfo)
        data = b"".join(stream_data)
        mock_object_info.size = len(data)
        return mock_object_info

    return get_object_info


def create_stream_getter(stream_data, chunk_size=5):
    """Create a stream getter function with range get capabilities. Simulates _get_object_stream"""

    def get_stream(start=None, end=None):
        if not start and not end:
            # Sequential reader case
            return iter(stream_data)
        else:
            # Range-based reader case:
            data = b"".join(stream_data)
            start_val = start or 0
            end_val = end if end is not None else len(data)
            data = data[start_val:end_val]
            # Split into chunks to simulate chunk-based stream
            chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]
            return iter(chunks)

    return get_stream


@composite
def bytestream_and_positions(draw):
    byte_array = draw(lists(binary(min_size=1, max_size=5000)))
    positions = draw(lists(integers(min_value=0, max_value=sum(map(len, byte_array)))))

    return byte_array, positions


@composite
def bytestream_and_position(draw, *, position_min_value: int = 0):
    if position_min_value > 0:
        byte_array = draw(lists(binary(min_size=1, max_size=5000), min_size=1))
    else:
        byte_array = draw(lists(binary(min_size=1, max_size=5000)))
    total_length = sum(map(len, byte_array))
    assume(total_length >= position_min_value)
    position = draw(integers(min_value=position_min_value, max_value=total_length))

    return byte_array, position


@pytest.mark.parametrize(
    "object_info, get_stream",
    [
        (None, lambda: None),
        (MOCK_OBJECT_INFO, lambda: None),
        (None, lambda: MOCK_STREAM),
        (MOCK_OBJECT_INFO, lambda: None),
        (MOCK_OBJECT_INFO, lambda: ""),
    ],
)
def test_s3reader_creation(
    reader_implementation: Type[S3Reader], object_info, get_stream
):
    s3reader = reader_implementation(
        TEST_BUCKET,
        TEST_KEY,
        lambda: object_info,
        get_stream,
    )
    assert s3reader
    assert s3reader.bucket == TEST_BUCKET
    assert s3reader.key == TEST_KEY
    assert s3reader._object_info == object_info
    assert s3reader._get_stream is get_stream


@pytest.mark.parametrize(
    "bucket, key",
    [(None, None), (None, ""), (None, TEST_KEY), ("", TEST_KEY)],
)
def test_s3reader_invalid_creation(reader_implementation: Type[S3Reader], bucket, key):
    with pytest.raises(ValueError, match="Bucket should be specified"):
        reader_implementation(bucket, key, lambda: None, lambda: [])


@pytest.mark.parametrize(
    "stream",
    [
        [b"1", b"2", b"3"],
        [],
        [b"hello!"],
    ],
)
def test_s3reader_read(reader_implementation: Type[S3Reader], stream):
    s3reader = create_s3reader(stream, reader_implementation)
    assert b"".join(stream) == s3reader.read()


@given(bytestream_and_positions())
def test_s3reader_seek(
    reader_implementation: Type[S3Reader],
    stream_and_positions: Tuple[List[bytes], List[int]],
):
    stream, positions = stream_and_positions
    s3reader = create_s3reader(stream, reader_implementation)

    bytesio = BytesIO(b"".join(stream))
    assert s3reader.tell() == 0

    for position in positions:
        s3reader.seek(position)
        bytesio.seek(position)

        assert s3reader.tell() == bytesio.tell() == position
        assert s3reader.read() == bytesio.read()
        assert s3reader.tell() == bytesio.tell()


@pytest.mark.parametrize("whence", [SEEK_SET, SEEK_CUR, SEEK_END])
def test_s3reader_seek_beyond_eof(reader_implementation: Type[S3Reader], whence):
    """Test seek beyond EOF clamps to object size correctly, for all 3 seek modes"""
    stream = [b"12345"]
    s3reader = create_s3reader(stream, reader_implementation)

    s3reader.seek(2)  # Position at 2

    # SEEK_SET: seek to 10; SEEK_CUR: seek to 12; SEEK_END: seek to 15
    pos = s3reader.seek(10, whence)

    # Should clamp to object size
    assert pos == 5
    assert s3reader.tell() == 5

    # All reader types should set _size correctly in all cases
    assert s3reader._size == 5


@pytest.mark.parametrize("whence", [SEEK_SET, SEEK_CUR, SEEK_END])
@given(bytestream_and_positions())
def test_s3reader_seek_beyond_eof_different_positions(
    whence,
    reader_implementation: Type[S3Reader],
    stream_and_positions: Tuple[List[bytes], List[int]],
):
    """
    Test seek beyond EOF clamps to object size correctly.
    Since we use `pos + stream_length + 1`, we will seek beyond eof for all 3 seek modes.
    """
    stream, positions = stream_and_positions
    stream_length = sum(map(len, stream))
    assume(stream_length > 0)

    for pos in positions:
        s3reader = create_s3reader(stream, reader_implementation)

        # +1 ensures beyond EOF, since we only get _size in sequential reader when reading beyond eof
        beyond_eof_pos = pos + stream_length + 1
        seek_return = s3reader.seek(beyond_eof_pos, whence)

        # Verify seek beyond EOF sets position to EOF
        assert s3reader.tell() == seek_return == stream_length
        assert s3reader._size == stream_length

        # Verify seeking back into file still works
        seek_return = s3reader.seek(pos)
        assert s3reader.tell() == seek_return == pos


@given(bytestream_and_positions())
def test_s3reader_read_with_sizes(
    reader_implementation: Type[S3Reader],
    stream_and_positions: Tuple[List[bytes], List[int]],
):
    stream, positions = stream_and_positions
    s3reader = create_s3reader(stream, reader_implementation)

    bytesio = BytesIO(b"".join(stream))

    positions.sort()
    for new_position in positions:
        size = new_position - bytesio.tell()

        assert s3reader.read(size) == bytesio.read(size)
        assert s3reader.tell() == bytesio.tell() == new_position


@given(
    lists(binary(min_size=1, max_size=5000)),
    integers(min_value=-sys.maxsize, max_value=-1),
)
def test_read_with_negative(
    reader_implementation: Type[S3Reader], stream: List[bytes], amount: int
):
    # Below -sys.maxsize, we get an OverflowError. I don't think it's too important to support this though.
    s3reader = create_s3reader(stream, reader_implementation)

    assert s3reader.read(amount) == b"".join(stream)


@pytest.mark.parametrize(
    "description, start, size, stream, expected_data, expected_position",
    [
        ("Zero-length read from start", 0, 0, [b"0123456789ABCDEF"], b"", 0),
        ("Zero-length read from middle", 5, 0, [b"0123456789ABCDEF"], b"", 5),
        ("Zero-length read from EOF", 16, 0, [b"0123456789ABCDEF"], b"", 16),
        ("Read near EOF", 10, 10, [b"0123456789ABCDEF"], b"ABCDEF", 16),
        ("Read beyond EOF", 16, 10, [b"0123456789ABCDEF"], b"", 16),
        ("Read from empty file", 0, 10, [], b"", 0),
        ("Seek beyond EOF then read", 20, 10, [b"0123456789ABCDEF"], b"", 16),
    ],
)
def test_s3reader_read_edge_cases(
    reader_implementation: Type[S3Reader],
    description,
    start,
    size,
    stream,
    expected_data,
    expected_position,
):
    """Test edge cases for S3Reader read method"""
    s3reader = create_s3reader(stream, reader_implementation)

    s3reader.seek(start)
    result = s3reader.read(size)
    assert result == expected_data
    assert s3reader.tell() == expected_position


@pytest.mark.parametrize(
    "description, start, buf_size, stream, expected_bytes_read, expected_position",
    [
        ("Zero-length readinto from start", 0, 0, [b"0123456789ABCDEF"], 0, 0),
        ("Zero-length readinto from middle", 5, 0, [b"0123456789ABCDEF"], 0, 5),
        ("Zero-length read from EOF", 16, 0, [b"0123456789ABCDEF"], 0, 16),
        ("Readinto near EOF", 10, 10, [b"0123456789ABCDEF"], 6, 16),
        ("Readinto beyond EOF", 16, 10, [b"0123456789ABCDEF"], 0, 16),
        ("Readinto from empty file", 0, 10, [], 0, 0),
        ("Seek beyond EOF then read", 20, 10, [b"0123456789ABCDEF"], 0, 16),
    ],
)
def test_s3reader_readinto_edge_cases(
    reader_implementation: Type[S3Reader],
    description,
    start,
    buf_size,
    stream,
    expected_bytes_read,
    expected_position,
):
    """Test edge cases for S3Reader readinto method"""
    s3reader = create_s3reader(stream, reader_implementation)

    s3reader.seek(start)
    buf = bytearray(buf_size)
    bytes_read = s3reader.readinto(buf)

    assert bytes_read == expected_bytes_read
    assert s3reader.tell() == expected_position

    if expected_bytes_read > 0:
        data = b"".join(stream)
        assert buf[:bytes_read] == data[start : start + bytes_read]


@given(
    lists(binary(min_size=1, max_size=5000)),
    integers(min_value=0, max_value=sys.maxsize),
)
def test_over_read(
    reader_implementation: Type[S3Reader], stream: List[bytes], overread: int
):
    # Currently fails when over sys.maxsize, but this number (~9 EB) is way bigger than the maximum S3 object size
    s3reader = create_s3reader(stream, reader_implementation)

    stream_length = sum(map(len, stream))
    to_read = stream_length + overread
    assume(to_read <= sys.maxsize)

    assert s3reader.read(to_read) == b"".join(stream)


def test_seeks_end(reader_implementation: Type[S3Reader]):
    s3reader = reader_implementation(
        TEST_BUCKET,
        TEST_KEY,
        lambda: None,
        lambda: iter([]),
    )
    s3reader._size = 10
    buf = memoryview(bytearray(10))

    assert s3reader.seek(0, SEEK_END) == 10
    assert s3reader.tell() == 10
    assert s3reader.read() == b""
    assert s3reader.readinto(buf) == 0

    assert s3reader.seek(0, SEEK_CUR) == 10
    assert s3reader.tell() == 10
    assert s3reader.read() == b""
    assert s3reader.readinto(buf) == 0


def test_seekable(reader_implementation: Type[S3Reader]):
    s3reader = reader_implementation(
        TEST_BUCKET,
        TEST_KEY,
        lambda: None,
        lambda: iter([]),
    )
    assert s3reader.seekable()


def test_readable(reader_implementation: Type[S3Reader]):
    s3reader = reader_implementation(
        TEST_BUCKET,
        TEST_KEY,
        lambda: None,
        lambda: iter([]),
    )
    assert s3reader.readable()


def test_not_writable(reader_implementation: Type[S3Reader]):
    s3reader = reader_implementation(
        TEST_BUCKET,
        TEST_KEY,
        lambda: None,
        lambda: iter([]),
    )
    assert not s3reader.writable()


@pytest.mark.parametrize(
    "whence, exception_type",
    [
        (5, ValueError),
        ("foo", TypeError),
    ],
)
def test_bad_whence(reader_implementation: Type[S3Reader], whence, exception_type):
    s3reader = reader_implementation(
        TEST_BUCKET,
        TEST_KEY,
        lambda: None,
        lambda: iter([]),
    )

    with pytest.raises(exception_type):
        s3reader.seek(0, whence)


@pytest.mark.parametrize(
    "offset",
    [0.4, 0.0, 1.0, "test", 1 + 2j, [1, 2, 3], {}, {2}],
)
def test_fails_with_non_int_arg(reader_implementation: Type[S3Reader], offset):
    s3reader = reader_implementation(
        TEST_BUCKET,
        TEST_KEY,
        lambda: None,
        lambda: iter([]),
    )

    with pytest.raises(TypeError):
        s3reader.seek(offset)
    with pytest.raises(TypeError):
        s3reader.read(offset)


@given(
    lists(binary(min_size=1, max_size=5000)),
    integers(max_value=-1),
)
def test_negative_seek(
    reader_implementation: Type[S3Reader], stream: List[bytes], seek: int
):
    s3reader = create_s3reader(stream, reader_implementation)

    with pytest.raises(ValueError):
        s3reader.seek(seek)


def test_end_seek_does_not_start_s3_request(reader_implementation: Type[S3Reader]):
    s3reader = reader_implementation(
        TEST_BUCKET,
        TEST_KEY,
        lambda: None,
        lambda: iter([]),
    )
    s3reader._size = 10
    s3reader.seek(0, SEEK_END)
    assert s3reader.tell() == 10


@given(bytestream_and_position(position_min_value=1))
def test_end_seek_with_offset(
    reader_implementation: Type[S3Reader], stream_and_positions: Tuple[List[bytes], int]
):
    stream, position = stream_and_positions
    s3reader = create_s3reader(stream, reader_implementation)

    s3reader._size = stream_length = sum(map(len, stream))

    s3reader.seek(-position, SEEK_END)
    assert s3reader.tell() == stream_length - position
    assert len(s3reader.read()) == position
    assert s3reader.tell() == stream_length


@given(bytestream_and_positions())
def test_s3reader_relative_seek(
    reader_implementation: Type[S3Reader],
    stream_and_positions: Tuple[List[bytes], List[int]],
):
    stream, positions = stream_and_positions
    s3reader = create_s3reader(stream, reader_implementation)

    bytesio = BytesIO(b"".join(stream))

    for new_position in positions:
        old_position = bytesio.tell()
        offset = new_position - old_position

        s3reader.seek(offset, SEEK_CUR)
        bytesio.seek(new_position)

        assert s3reader.tell() == bytesio.tell() == new_position
        assert s3reader.read() == bytesio.read()

        s3reader.seek(new_position)
        bytesio.seek(new_position)


@given(
    lists(binary(min_size=1, max_size=5000)),
)
def test_s3reader_writes_size_after_read_all(
    reader_implementation: Type[S3Reader], stream: List[bytes]
):
    s3reader = create_s3reader(stream, reader_implementation)

    assert s3reader._size is None
    s3reader.read()
    assert s3reader._size == sum(map(len, stream))


@given(
    lists(binary(min_size=20, max_size=30), min_size=0, max_size=2),
    integers(min_value=0, max_value=10),
)
def test_s3reader_readinto_buffer_smaller_than_chunks(
    reader_implementation: Type[S3Reader], stream: List[bytes], buf_size: int
):
    s3reader = create_s3reader(stream, reader_implementation)

    assert s3reader._size is None
    total_length = sum(map(len, stream))
    buf = memoryview(bytearray(buf_size))
    # We're able to read all the available data or the data that can be accommodated in buf
    if buf_size > 0 and total_length > 0:
        assert s3reader.readinto(buf) == buf_size
        assert s3reader.tell() == buf_size
        # confirm that read data is the same as in source
        assert buf[:buf_size] == (b"".join(stream))[:buf_size]
    else:
        assert s3reader.readinto(buf) == 0
        assert s3reader.tell() == 0


@given(
    lists(binary(min_size=20, max_size=30), min_size=2, max_size=3),
    integers(min_value=30, max_value=40),
)
def test_s3reader_readinto_buffer_bigger_than_chunks(
    reader_implementation: Type[S3Reader], stream: List[bytes], buf_size: int
):
    s3reader = create_s3reader(stream, reader_implementation)

    assert s3reader._size is None
    buf = memoryview(bytearray(buf_size))
    # We're able to read the data that can be accommodated in buf
    assert s3reader.readinto(buf) == buf_size
    assert s3reader.tell() == buf_size
    all_data = b"".join(stream)
    # confirm that read data is the same as in source
    assert buf == all_data[:buf_size]


@given(
    lists(binary(min_size=20, max_size=30), min_size=1, max_size=3),
    integers(min_value=100, max_value=100),
)
def test_s3reader_readinto_buffer_bigger_than_whole_object(
    reader_implementation: Type[S3Reader], stream: List[bytes], buf_size: int
):
    s3reader = create_s3reader(stream, reader_implementation)

    assert s3reader._size is None
    total_length = sum(map(len, stream))
    buf = memoryview(bytearray(buf_size))
    # We're able to read all the available data
    assert s3reader.readinto(buf) == total_length
    assert s3reader.tell() == total_length
    all_data = b"".join(stream)
    # confirm that read data is the same as in source
    assert buf[:total_length] == all_data
    assert s3reader._size == total_length


@given(
    lists(binary(min_size=2, max_size=12), min_size=1, max_size=5),
    integers(min_value=3, max_value=10),
    integers(min_value=0, max_value=1),
)
def test_s3reader_mixing_readinto_and_read(
    reader_implementation: Type[S3Reader], stream: List[bytes], buf_size: int, flip: int
):
    position = 0
    loops_count = 20
    all_data = b"".join(stream)
    total_length = len(all_data)
    buf = memoryview(bytearray(buf_size))
    s3reader = create_s3reader(stream, reader_implementation)

    for i in range(0, loops_count):
        if position >= total_length:
            break

        if (i + flip) % 2 == 0:
            result = s3reader.read(buf_size)
            # confirm that read data is the same as in source
            if position + buf_size < total_length:
                assert result[:buf_size] == all_data[position : position + buf_size]
            else:
                read_bytes = total_length - position
                assert result[:read_bytes] == all_data[position:total_length]
            position += buf_size
        else:
            read_bytes = s3reader.readinto(buf)
            # confirm that read data is the same as in source
            assert buf[position:read_bytes] == all_data[position:read_bytes]
            position += read_bytes

        if position > total_length:
            # we read all the data, it is time to stop
            assert s3reader.tell() == total_length
            break
        else:
            # confirm that position is as expected
            assert s3reader.tell() == position
