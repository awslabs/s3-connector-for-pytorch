import logging
import sys
from io import BytesIO, SEEK_END, SEEK_CUR
from typing import List, Tuple
from unittest.mock import Mock

import pytest
from hypothesis import given, assume
from hypothesis.strategies import lists, binary, integers, composite
from s3dataset_s3_client._s3dataset import ObjectInfo, GetObjectStream

from s3dataset_s3_client import S3Object

logging.basicConfig(
    format="%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s"
)
logging.getLogger().setLevel(1)

log = logging.getLogger(__name__)


TEST_BUCKET = "test-bucket"
TEST_KEY = "test-key"
MOCK_OBJECT_INFO = Mock(ObjectInfo)
MOCK_STREAM = Mock(GetObjectStream)


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
def test_s3object_creation(object_info, get_stream):
    s3object = S3Object(TEST_BUCKET, TEST_KEY, object_info, get_stream)
    assert s3object
    assert s3object.bucket == TEST_BUCKET
    assert s3object.key == TEST_KEY
    assert s3object.object_info == object_info
    assert s3object._get_stream is get_stream


@pytest.mark.parametrize(
    "bucket, key",
    [(None, None), (None, ""), (None, TEST_KEY), ("", TEST_KEY)],
)
def test_s3object_invalid_creation(bucket, key):
    with pytest.raises(ValueError) as error:
        S3Object(bucket, key)
    assert str(error.value) == "Bucket should be specified"


@pytest.mark.parametrize(
    "stream",
    [
        [b"1", b"2", b"3"],
        [],
        [b"hello!"],
    ],
)
def test_s3object_prefetch(stream):
    s3object = S3Object(TEST_BUCKET, TEST_KEY, None, lambda: stream)
    assert s3object._stream is None
    s3object.prefetch()
    assert s3object._stream is stream
    s3object.prefetch()
    assert s3object._stream is stream


@pytest.mark.parametrize(
    "stream",
    [
        [b"1", b"2", b"3"],
        [],
        [b"hello!"],
    ],
)
def test_s3object_read(stream):
    s3object = S3Object(TEST_BUCKET, TEST_KEY, None, lambda: stream)
    assert s3object._stream is None
    assert b"".join(stream) == s3object.read()


@given(bytestream_and_positions())
def test_s3object_seek(stream_and_positions: Tuple[List[bytes], List[int]]):
    stream, positions = stream_and_positions
    s3object = S3Object(TEST_BUCKET, TEST_KEY, None, lambda: iter(stream))
    bytesio = BytesIO(b"".join(stream))
    assert s3object.tell() == 0

    for position in positions:
        s3object.seek(position)
        bytesio.seek(position)

        assert s3object.tell() == s3object._buffer.tell() == bytesio.tell() == position
        assert s3object._buffer_size() >= position
        assert s3object.read() == bytesio.read()
        assert s3object.tell() == s3object._buffer.tell() == bytesio.tell()


@given(bytestream_and_positions())
def test_s3object_read(stream_and_positions: Tuple[List[bytes], List[int]]):
    stream, positions = stream_and_positions
    s3object = S3Object(TEST_BUCKET, TEST_KEY, None, lambda: iter(stream))
    bytesio = BytesIO(b"".join(stream))

    positions.sort()
    for new_position in positions:
        size = new_position - bytesio.tell()

        assert s3object.read(size) == bytesio.read(size)
        assert (
            s3object.tell() == s3object._buffer.tell() == bytesio.tell() == new_position
        )


@pytest.mark.parametrize(
    "stream, to_read",
    [
        ([b"1"], 0),
        ([b"1", b"2", b"3"], 1),
    ],
)
def test_s3object_does_not_buffer_all(stream: List[bytes], to_read: int):
    s3object = S3Object(TEST_BUCKET, TEST_KEY, None, lambda: iter(stream))

    assert len(s3object.read(to_read)) == to_read
    assert s3object._stream is None or list(s3object._stream) != []


@given(
    lists(binary(min_size=1, max_size=5000)),
    integers(min_value=-sys.maxsize, max_value=-1),
)
def test_read_with_negative(stream: List[bytes], amount: int):
    # Below -sys.maxsize, we get an OverflowError. I don't think it's too important to support this though.
    s3object = S3Object(TEST_BUCKET, TEST_KEY, None, lambda: iter(stream))
    assert s3object.read(amount) == b"".join(stream)


@given(
    lists(binary(min_size=1, max_size=5000)),
    integers(min_value=0, max_value=sys.maxsize),
)
def test_over_read(stream: List[bytes], overread: int):
    # Currently fails when over sys.maxsize, but this number (~9 EB) is way bigger than the maximum S3 object size
    s3object = S3Object(TEST_BUCKET, TEST_KEY, None, lambda: iter(stream))
    stream_length = sum(map(len, stream))
    to_read = stream_length + overread
    assume(to_read <= sys.maxsize)

    assert s3object.read(to_read) == b"".join(stream)


def test_seeks_end():
    s3object = S3Object(TEST_BUCKET, TEST_KEY, None, lambda: iter([]))
    s3object._size = 10

    assert s3object.seek(0, SEEK_END) == 10
    assert s3object.tell() == 10
    assert s3object.read() == b""

    assert s3object.seek(0, SEEK_CUR) == 10
    assert s3object.tell() == 10
    assert s3object.read() == b""


def test_not_writable():
    s3object = S3Object(TEST_BUCKET, TEST_KEY, None, lambda: iter([]))
    assert not s3object.writable()


def test_bad_whence():
    s3object = S3Object(TEST_BUCKET, TEST_KEY, None, lambda: iter([]))

    with pytest.raises(TypeError):
        s3object.seek(0, "foo")


@pytest.mark.parametrize(
    "offset",
    [0.4, 0.0, 1.0, "test", 1 + 2j, [1, 2, 3], {}, {2}],
)
def test_fails_with_non_int_arg(offset):
    s3object = S3Object(TEST_BUCKET, TEST_KEY, None, lambda: iter([]))

    with pytest.raises(TypeError):
        s3object.seek(offset)
    with pytest.raises(TypeError):
        s3object.read(offset)


@given(
    lists(binary(min_size=1, max_size=5000)),
    integers(max_value=-1),
)
def test_negative_seek(stream: List[bytes], seek: int):
    s3object = S3Object(TEST_BUCKET, TEST_KEY, None, lambda: iter(stream))
    with pytest.raises(ValueError):
        s3object.seek(seek)


def test_end_seek_does_not_start_s3_request():
    s3object = S3Object(TEST_BUCKET, TEST_KEY, None, lambda: iter([]))
    s3object._size = 10
    s3object.seek(0, SEEK_END)
    assert s3object.tell() == 10
    assert s3object._stream is None


@given(bytestream_and_position(position_min_value=1))
def test_end_seek_with_offset(stream_and_positions: Tuple[List[bytes], int]):
    stream, position = stream_and_positions
    s3object = S3Object(TEST_BUCKET, TEST_KEY, None, lambda: iter(stream))
    s3object._size = stream_length = sum(map(len, stream))

    s3object.seek(-position, SEEK_END)
    assert s3object.tell() == stream_length - position
    assert len(s3object.read()) == position
    assert s3object.tell() == stream_length


@given(bytestream_and_positions())
def test_s3object_relative_seek(stream_and_positions: Tuple[List[bytes], List[int]]):
    stream, positions = stream_and_positions
    s3object = S3Object(TEST_BUCKET, TEST_KEY, None, lambda: iter(stream))
    bytesio = BytesIO(b"".join(stream))

    for new_position in positions:
        old_position = bytesio.tell()
        offset = new_position - old_position

        s3object.seek(offset, SEEK_CUR)
        bytesio.seek(new_position)

        assert (
            s3object.tell() == s3object._buffer.tell() == bytesio.tell() == new_position
        )
        assert s3object.read() == bytesio.read()

        s3object.seek(new_position)
        bytesio.seek(new_position)


@given(
    lists(binary(min_size=1, max_size=5000)),
)
def test_s3object_writes_size_after_read_all(stream: List[bytes]):
    s3object = S3Object(TEST_BUCKET, TEST_KEY, None, lambda: iter(stream))
    assert s3object._size is None
    s3object.read()
    assert s3object._size == sum(map(len, stream))


@given(
    lists(binary(min_size=1, max_size=5000)),
)
def test_s3object_writes_size_after_read_all_explicit(stream: List[bytes]):
    s3object = S3Object(TEST_BUCKET, TEST_KEY, None, lambda: iter(stream))
    assert s3object._size is None
    total_length = sum(map(len, stream))
    # We're able to read all the data
    assert len(s3object.read(total_length)) == total_length
    # We don't know we've reached the end
    assert s3object._size is None
    # Reading past the end gives us empty
    assert s3object.read(1) == b""
    # Once we've read past the end, we know how big the file is
    assert s3object._size == total_length
