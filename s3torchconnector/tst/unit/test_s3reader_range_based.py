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

from s3torchconnector import S3Reader, S3ReaderConfig
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

RANGE_BASED_READER_CONFIG = S3ReaderConfig(
    reader_type=S3ReaderConfig.ReaderType.RANGE_BASED
)


def create_range_s3reader(stream):
    return S3Reader(
        TEST_BUCKET,
        TEST_KEY,
        create_object_info_getter(stream),
        create_stream_getter(stream),
        reader_config=RANGE_BASED_READER_CONFIG,
    )


@given(lists(binary(min_size=1, max_size=5000)))
def test_s3reader_writes_size_before_read_all(stream):
    s3reader = create_range_s3reader(stream)
    assert s3reader._reader._size is None
    total_length = sum(map(len, stream))
    # We're able to read all the data
    assert len(s3reader.read(total_length)) == total_length
    # Read operation writes size before reading
    assert s3reader._reader._size == total_length
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
    assert s3reader._reader._size is None
    total_length = sum(map(len, stream))
    buf = memoryview(bytearray(buf_size))
    # We're able to read all the available data or the data that can be accommodated in buf
    if buf_size > 0 and total_length > 0:
        assert s3reader.readinto(buf) == buf_size
        assert s3reader.tell() == buf_size
        # Readinto operation does write size
        assert s3reader._reader._size == total_length
        # confirm that read data is the same as in source
        assert buf[:buf_size] == (b"".join(stream))[:buf_size]
    else:
        assert s3reader.readinto(buf) == 0
        assert s3reader.tell() == 0
