#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import logging
import sys
from io import BytesIO, SEEK_SET, SEEK_END, SEEK_CUR
from typing import List, Tuple
from unittest.mock import Mock

import pytest
from hypothesis import given, assume
from hypothesis.strategies import lists, binary, integers, composite
from s3torchconnectorclient._mountpoint_s3_client import ObjectInfo, GetObjectStream

from s3torchconnector import S3Reader, S3ReaderConfig
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
READER_TYPE_TO_CLASS = {
    S3ReaderConfig.ReaderType.SEQUENTIAL: SequentialS3Reader,
    S3ReaderConfig.ReaderType.RANGE_BASED: RangedS3Reader,
}


@pytest.fixture(
    params=[
        S3ReaderConfig(reader_type=S3ReaderConfig.ReaderType.SEQUENTIAL),
        S3ReaderConfig(reader_type=S3ReaderConfig.ReaderType.RANGE_BASED),
    ],
    scope="module",
)
def reader_config(request) -> S3ReaderConfig:
    """Provide S3ReaderConfig instances for all supported reader types."""
    return request.param


def create_s3reader(stream, reader_config: S3ReaderConfig):
    return S3Reader(
        TEST_BUCKET,
        TEST_KEY,
        create_object_info_getter(stream),
        create_stream_getter(stream),
        reader_config=reader_config,
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


def test_s3reader_reader_type_instance(reader_config: S3ReaderConfig):
    """Test that the reader type is set correctly"""
    stream = [b"test"]
    s3reader = create_s3reader(stream, reader_config)
    assert isinstance(s3reader, READER_TYPE_TO_CLASS[reader_config.reader_type])


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
def test_s3reader_creation(reader_config: S3ReaderConfig, object_info, get_stream):
    s3reader = S3Reader(
        TEST_BUCKET,
        TEST_KEY,
        lambda: object_info,
        get_stream,
        reader_config=reader_config,
    )
    assert s3reader
    assert s3reader.bucket == TEST_BUCKET
    assert s3reader.key == TEST_KEY
    assert s3reader._object_info == object_info
    assert s3reader._get_stream is get_stream
    assert isinstance(s3reader, READER_TYPE_TO_CLASS[reader_config.reader_type])


@pytest.mark.parametrize(
    "bucket, key",
    [(None, None), (None, ""), (None, TEST_KEY), ("", TEST_KEY)],
)
def test_s3reader_invalid_creation(reader_config: S3ReaderConfig, bucket, key):
    with pytest.raises(ValueError, match="Bucket should be specified"):
        S3Reader(bucket, key, lambda: None, lambda: [], reader_config=reader_config)


@pytest.mark.parametrize(
    "invalid_config",
    [
        42,
        "sequential",
        {"reader_type": "sequential"},
        True,
        None.__class__,  # NoneType
    ],
)
def test_s3reader_invalid_config_type(invalid_config):
    """Test that S3Reader raises TypeError for invalid reader_config types"""
    with pytest.raises(
        TypeError, match="reader_config must be an instance of S3ReaderConfig"
    ):
        S3Reader(
            TEST_BUCKET,
            TEST_KEY,
            lambda: None,
            lambda: None,
            reader_config=invalid_config,
        )


@pytest.mark.parametrize(
    "factory_method, expected_type",
    [
        (S3ReaderConfig.sequential, S3ReaderConfig.ReaderType.SEQUENTIAL),
        (S3ReaderConfig.range_based, S3ReaderConfig.ReaderType.RANGE_BASED),
    ],
)
def test_s3readerconfig_alternative_constructors(factory_method, expected_type):
    """Test alternative constructor methods for creating S3ReaderConfig instances."""
    config = factory_method()
    assert isinstance(config, S3ReaderConfig)
    assert config.reader_type == expected_type

    # Test alternative construcotr methods returns same config as original method
    assert config == S3ReaderConfig(reader_type=expected_type)

    # Test alternative constructor methods create new instances
    assert factory_method() is not factory_method()

    # Test with S3Reader
    stream = [b"test"]
    s3reader = create_s3reader(stream, factory_method())
    assert isinstance(s3reader, READER_TYPE_TO_CLASS[expected_type])


@pytest.mark.parametrize(
    "stream",
    [
        [b"1", b"2", b"3"],
        [],
        [b"hello!"],
    ],
)
def test_s3reader_read(reader_config: S3ReaderConfig, stream):
    s3reader = create_s3reader(stream, reader_config)
    assert isinstance(s3reader, READER_TYPE_TO_CLASS[reader_config.reader_type])
    assert s3reader._stream is None
    assert b"".join(stream) == s3reader.read()


@given(bytestream_and_positions())
def test_s3reader_seek(
    reader_config: S3ReaderConfig, stream_and_positions: Tuple[List[bytes], List[int]]
):
    stream, positions = stream_and_positions
    s3reader = create_s3reader(stream, reader_config)
    assert isinstance(s3reader, READER_TYPE_TO_CLASS[reader_config.reader_type])
    bytesio = BytesIO(b"".join(stream))
    assert s3reader.tell() == 0

    for position in positions:
        s3reader.seek(position)
        bytesio.seek(position)

        assert s3reader.tell() == bytesio.tell() == position
        assert s3reader.read() == bytesio.read()
        assert s3reader.tell() == bytesio.tell()


@given(bytestream_and_positions())
def test_s3reader_read_with_sizes(
    reader_config: S3ReaderConfig, stream_and_positions: Tuple[List[bytes], List[int]]
):
    stream, positions = stream_and_positions
    s3reader = create_s3reader(stream, reader_config)
    assert isinstance(s3reader, READER_TYPE_TO_CLASS[reader_config.reader_type])
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
    reader_config: S3ReaderConfig, stream: List[bytes], amount: int
):
    # Below -sys.maxsize, we get an OverflowError. I don't think it's too important to support this though.
    s3reader = create_s3reader(stream, reader_config)
    assert isinstance(s3reader, READER_TYPE_TO_CLASS[reader_config.reader_type])
    assert s3reader.read(amount) == b"".join(stream)


@pytest.mark.parametrize(
    "description, start, size, stream, expected_data, expected_position",
    [
        ("Zero-length read from start", 0, 0, [b"0123456789ABCDEF"], b"", 0),
        ("Zero-length read from middle", 5, 0, [b"0123456789ABCDEF"], b"", 5),
        ("Read near EOF", 10, 10, [b"0123456789ABCDEF"], b"ABCDEF", 16),
        ("Read beyond EOF", 16, 10, [b"0123456789ABCDEF"], b"", 16),
        ("Read from empty file", 0, 10, [], b"", 0),
    ],
)
def test_s3reader_read_edge_cases(
    reader_config: S3ReaderConfig,
    description,
    start,
    size,
    stream,
    expected_data,
    expected_position,
):
    """Test edge cases for S3Reader read method"""
    s3reader = create_s3reader(stream, reader_config)
    assert isinstance(s3reader, READER_TYPE_TO_CLASS[reader_config.reader_type])
    s3reader.seek(start)
    result = s3reader.read(size)
    assert result == expected_data
    assert s3reader.tell() == expected_position


@pytest.mark.parametrize(
    "description, start, buf_size, stream, expected_bytes_read, expected_position",
    [
        ("Zero-length readinto from start", 0, 0, [b"0123456789ABCDEF"], 0, 0),
        ("Zero-length readinto from middle", 5, 0, [b"0123456789ABCDEF"], 0, 5),
        ("Readinto near EOF", 10, 10, [b"0123456789ABCDEF"], 6, 16),
        ("Readinto beyond EOF", 16, 10, [b"0123456789ABCDEF"], 0, 16),
        ("Readinto from empty file", 0, 10, [], 0, 0),
    ],
)
def test_s3reader_readinto_edge_cases(
    reader_config: S3ReaderConfig,
    description,
    start,
    buf_size,
    stream,
    expected_bytes_read,
    expected_position,
):
    """Test edge cases for S3Reader readinto method"""
    s3reader = create_s3reader(stream, reader_config)
    assert isinstance(s3reader, READER_TYPE_TO_CLASS[reader_config.reader_type])

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
def test_over_read(reader_config: S3ReaderConfig, stream: List[bytes], overread: int):
    # Currently fails when over sys.maxsize, but this number (~9 EB) is way bigger than the maximum S3 object size
    s3reader = create_s3reader(stream, reader_config)
    assert isinstance(s3reader, READER_TYPE_TO_CLASS[reader_config.reader_type])
    stream_length = sum(map(len, stream))
    to_read = stream_length + overread
    assume(to_read <= sys.maxsize)

    assert s3reader.read(to_read) == b"".join(stream)


def test_seeks_end(reader_config: S3ReaderConfig):
    s3reader = S3Reader(
        TEST_BUCKET,
        TEST_KEY,
        lambda: None,
        lambda: iter([]),
        reader_config=reader_config,
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


def test_not_writable(reader_config: S3ReaderConfig):
    s3reader = S3Reader(
        TEST_BUCKET,
        TEST_KEY,
        lambda: None,
        lambda: iter([]),
        reader_config=reader_config,
    )
    assert not s3reader.writable()


@pytest.mark.parametrize(
    "whence, exception_type",
    [
        (5, ValueError),
        ("foo", TypeError),
    ],
)
def test_bad_whence(reader_config: S3ReaderConfig, whence, exception_type):
    s3reader = S3Reader(
        TEST_BUCKET,
        TEST_KEY,
        lambda: None,
        lambda: iter([]),
        reader_config=reader_config,
    )

    with pytest.raises(exception_type):
        s3reader.seek(0, whence)


@pytest.mark.parametrize(
    "offset",
    [0.4, 0.0, 1.0, "test", 1 + 2j, [1, 2, 3], {}, {2}],
)
def test_fails_with_non_int_arg(reader_config: S3ReaderConfig, offset):
    s3reader = S3Reader(
        TEST_BUCKET,
        TEST_KEY,
        lambda: None,
        lambda: iter([]),
        reader_config=reader_config,
    )

    with pytest.raises(TypeError):
        s3reader.seek(offset)
    with pytest.raises(TypeError):
        s3reader.read(offset)


@given(
    lists(binary(min_size=1, max_size=5000)),
    integers(max_value=-1),
)
def test_negative_seek(reader_config: S3ReaderConfig, stream: List[bytes], seek: int):
    s3reader = create_s3reader(stream, reader_config)
    assert isinstance(s3reader, READER_TYPE_TO_CLASS[reader_config.reader_type])
    with pytest.raises(ValueError):
        s3reader.seek(seek)


def test_end_seek_does_not_start_s3_request(reader_config: S3ReaderConfig):
    s3reader = S3Reader(
        TEST_BUCKET,
        TEST_KEY,
        lambda: None,
        lambda: iter([]),
        reader_config=reader_config,
    )
    s3reader._size = 10
    s3reader.seek(0, SEEK_END)
    assert s3reader.tell() == 10
    assert s3reader._stream is None


@given(bytestream_and_position(position_min_value=1))
def test_end_seek_with_offset(
    reader_config: S3ReaderConfig, stream_and_positions: Tuple[List[bytes], int]
):
    stream, position = stream_and_positions
    s3reader = create_s3reader(stream, reader_config)
    assert isinstance(s3reader, READER_TYPE_TO_CLASS[reader_config.reader_type])
    s3reader._size = stream_length = sum(map(len, stream))

    s3reader.seek(-position, SEEK_END)
    assert s3reader.tell() == stream_length - position
    assert len(s3reader.read()) == position
    assert s3reader.tell() == stream_length


@given(bytestream_and_positions())
def test_s3reader_relative_seek(
    reader_config: S3ReaderConfig, stream_and_positions: Tuple[List[bytes], List[int]]
):
    stream, positions = stream_and_positions
    s3reader = create_s3reader(stream, reader_config)
    assert isinstance(s3reader, READER_TYPE_TO_CLASS[reader_config.reader_type])
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
    reader_config: S3ReaderConfig, stream: List[bytes]
):
    s3reader = create_s3reader(stream, reader_config)
    assert isinstance(s3reader, READER_TYPE_TO_CLASS[reader_config.reader_type])
    assert s3reader._size is None
    s3reader.read()
    assert s3reader._size == sum(map(len, stream))


@given(
    lists(binary(min_size=20, max_size=30), min_size=0, max_size=2),
    integers(min_value=0, max_value=10),
)
def test_s3reader_readinto_buffer_smaller_than_chunks(
    reader_config: S3ReaderConfig, stream: List[bytes], buf_size: int
):
    s3reader = create_s3reader(stream, reader_config)
    assert isinstance(s3reader, READER_TYPE_TO_CLASS[reader_config.reader_type])
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
    reader_config: S3ReaderConfig, stream: List[bytes], buf_size: int
):
    s3reader = create_s3reader(stream, reader_config)
    assert isinstance(s3reader, READER_TYPE_TO_CLASS[reader_config.reader_type])
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
    reader_config: S3ReaderConfig, stream: List[bytes], buf_size: int
):
    s3reader = create_s3reader(stream, reader_config)
    assert isinstance(s3reader, READER_TYPE_TO_CLASS[reader_config.reader_type])
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
    reader_config: S3ReaderConfig, stream: List[bytes], buf_size: int, flip: int
):
    position = 0
    loops_count = 20
    all_data = b"".join(stream)
    total_length = len(all_data)
    buf = memoryview(bytearray(buf_size))
    s3reader = create_s3reader(stream, reader_config)
    assert isinstance(s3reader, READER_TYPE_TO_CLASS[reader_config.reader_type])
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
