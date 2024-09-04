#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from io import BytesIO
from typing import List, Tuple
from unittest.mock import Mock

import pytest
from hypothesis import given
from hypothesis.strategies import lists, binary, composite
from s3torchconnectorclient._mountpoint_s3_client import ObjectInfo, PutObjectStream

from s3torchconnector import S3Writer

MOCK_OBJECT_INFO = Mock(ObjectInfo)
MOCK_STREAM = Mock(PutObjectStream)


@composite
def bytestream_and_lengths(draw):
    byte_array = draw(lists(binary(min_size=1, max_size=5000)))
    lengths = [len(b) for b in byte_array]
    return byte_array, lengths


def test_s3writer_creation():
    s3writer = S3Writer(MOCK_STREAM)
    assert s3writer
    assert isinstance(s3writer.stream, PutObjectStream)


@pytest.mark.parametrize(
    "stream",
    [
        [b"1", b"2", b"3"],
        [],
        [b"hello!"],
    ],
)
def test_s3writer_write(stream):
    s3writer = S3Writer(MOCK_STREAM)
    s3writer.write(stream)
    s3writer.close()
    MOCK_STREAM.write.assert_called_with(stream)


@given(bytestream_and_lengths())
def test_s3writer_tell(stream_and_lengths: Tuple[List[bytes], List[int]]):
    with S3Writer(MOCK_STREAM) as s3writer, BytesIO() as bytewriter:
        for data, length in zip(*stream_and_lengths):
            b_length = s3writer.write(data)
            bytewriter.write(data)

            assert b_length == length
            assert bytewriter.tell() == s3writer.tell()
