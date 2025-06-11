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
from s3torchconnectorclient._mountpoint_s3_client import ObjectInfo, GetObjectStream

from s3torchconnector import S3Reader, ReaderType
from .test_s3reader import (
    TEST_BUCKET,
    TEST_KEY,
    MOCK_OBJECT_INFO,
    MOCK_STREAM,
    bytestream_and_positions,
    bytestream_and_position,
)

logging.basicConfig(
    format="%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s"
)
logging.getLogger().setLevel(1)

log = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def reader_type():
    return ReaderType.SEQUENTIAL


@pytest.mark.parametrize(
    "stream",
    [
        [b"1", b"2", b"3"],
        [],
        [b"hello!"],
    ],
)
def test_s3reader_prefetch(stream, reader_type: ReaderType):
    s3reader = S3Reader(
        TEST_BUCKET, TEST_KEY, lambda: None, lambda: stream, reader_type=reader_type
    )
    assert s3reader._reader._stream is None
    s3reader.prefetch()
    assert s3reader._reader._stream is stream
    s3reader.prefetch()
    assert s3reader._reader._stream is stream


@given(bytestream_and_positions())
def test_s3reader_updates_buffer_position_during_sized_reads(
    reader_type: ReaderType, stream_and_positions: Tuple[List[bytes], List[int]]
):
    stream, positions = stream_and_positions
    s3reader = S3Reader(
        TEST_BUCKET,
        TEST_KEY,
        lambda: None,
        lambda: iter(stream),
        reader_type=reader_type,
    )
    bytesio = BytesIO(b"".join(stream))

    positions.sort()
    for new_position in positions:
        size = new_position - bytesio.tell()

        assert s3reader.read(size) == bytesio.read(size)
        assert (
            s3reader.tell()
            == s3reader._reader._buffer.tell()
            == bytesio.tell()
            == new_position
        )


@given(bytestream_and_positions())
def test_s3reader_updates_buffer_position_during_seek(
    reader_type: ReaderType, stream_and_positions: Tuple[List[bytes], List[int]]
):
    stream, positions = stream_and_positions
    s3reader = S3Reader(
        TEST_BUCKET,
        TEST_KEY,
        lambda: None,
        lambda: iter(stream),
        reader_type=reader_type,
    )
    bytesio = BytesIO(b"".join(stream))
    assert s3reader.tell() == 0

    for position in positions:
        s3reader.seek(position)
        bytesio.seek(position)

        assert (
            s3reader.tell()
            == s3reader._reader._buffer.tell()
            == bytesio.tell()
            == position
        )
        assert s3reader._reader._buffer_size() >= position
        assert s3reader.read() == bytesio.read()
        assert s3reader.tell() == s3reader._reader._buffer.tell() == bytesio.tell()


@given(bytestream_and_positions())
def test_s3reader_updates_buffer_position_during_relative_seek(
    reader_type: ReaderType, stream_and_positions: Tuple[List[bytes], List[int]]
):
    stream, positions = stream_and_positions
    s3reader = S3Reader(
        TEST_BUCKET,
        TEST_KEY,
        lambda: None,
        lambda: iter(stream),
        reader_type=reader_type,
    )
    bytesio = BytesIO(b"".join(stream))

    for new_position in positions:
        old_position = bytesio.tell()
        offset = new_position - old_position

        s3reader.seek(offset, SEEK_CUR)
        bytesio.seek(new_position)

        assert (
            s3reader.tell()
            == s3reader._reader._buffer.tell()
            == bytesio.tell()
            == new_position
        )
        assert s3reader.read() == bytesio.read()

        s3reader.seek(new_position)
        bytesio.seek(new_position)


@pytest.mark.parametrize(
    "stream, to_read",
    [
        ([b"1"], 0),
        ([b"1", b"2", b"3"], 1),
    ],
)
def test_s3reader_does_not_buffer_all(
    reader_type: ReaderType, stream: List[bytes], to_read: int
):
    s3reader = S3Reader(
        TEST_BUCKET,
        TEST_KEY,
        lambda: None,
        lambda: iter(stream),
        reader_type=reader_type,
    )

    assert len(s3reader.read(to_read)) == to_read
    assert s3reader._reader._stream is None or list(s3reader._reader._stream) != []


@given(
    lists(binary(min_size=1, max_size=5000)),
)
def test_s3reader_writes_size_after_read_all_explicit(
    reader_type: ReaderType, stream: List[bytes]
):
    s3reader = S3Reader(
        TEST_BUCKET,
        TEST_KEY,
        lambda: None,
        lambda: iter(stream),
        reader_type=reader_type,
    )
    assert s3reader._reader._size is None
    total_length = sum(map(len, stream))
    # We're able to read all the data
    assert len(s3reader.read(total_length)) == total_length
    # We don't know we've reached the end
    assert s3reader._reader._size is None
    # Reading past the end gives us empty
    assert s3reader.read(1) == b""
    # Once we've read past the end, we know how big the file is
    assert s3reader._reader._size == total_length


@given(
    lists(binary(min_size=20, max_size=30), min_size=0, max_size=2),
    integers(min_value=0, max_value=10),
)
def test_s3reader_does_not_write_size_when_readinto_buffer_smaller_than_chunks(
    reader_type: ReaderType, stream: List[bytes], buf_size: int
):
    s3reader = S3Reader(
        TEST_BUCKET,
        TEST_KEY,
        lambda: None,
        lambda: iter(stream),
        reader_type=reader_type,
    )
    assert s3reader._reader._size is None
    total_length = sum(map(len, stream))
    buf = memoryview(bytearray(buf_size))
    # We're able to read all the available data or the data that can be accommodated in buf
    if buf_size > 0 and total_length > 0:
        assert s3reader.readinto(buf) == buf_size
        assert s3reader.tell() == buf_size
        # We haven't reached the end yet, so we don't write size
        assert s3reader._reader._size is None
        # confirm that read data is the same as in source
        assert buf[:buf_size] == (b"".join(stream))[:buf_size]
    else:
        assert s3reader.readinto(buf) == 0
        assert s3reader.tell() == 0


@given(
    lists(binary(min_size=20, max_size=30), min_size=1, max_size=3),
    integers(min_value=100, max_value=100),
)
def test_s3reader_writes_size_when_readinto_buffer_bigger_than_whole_object(
    reader_type: ReaderType, stream: List[bytes], buf_size: int
):
    s3reader = S3Reader(
        TEST_BUCKET,
        TEST_KEY,
        lambda: None,
        lambda: iter(stream),
        reader_type=reader_type,
    )
    assert s3reader._reader._size is None
    total_length = sum(map(len, stream))
    buf = memoryview(bytearray(buf_size))
    # We're able to read all the available data
    assert s3reader.readinto(buf) == total_length
    assert s3reader.tell() == total_length
    all_data = b"".join(stream)
    # confirm that read data is the same as in source
    assert buf[:total_length] == all_data
    # confirm that size is written
    assert s3reader._reader._size == total_length
    # confirm that we've reached the end
    assert s3reader.read() == b""
