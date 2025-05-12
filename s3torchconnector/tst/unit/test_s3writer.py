#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from io import BytesIO
from typing import List, Tuple
from unittest.mock import Mock
import threading

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


def test_multiple_close_calls():
    """Test that multiple calls to close() only close the stream once."""
    MOCK_STREAM.reset_mock()

    writer = S3Writer(MOCK_STREAM)

    writer.close()
    writer.close()
    writer.close()

    MOCK_STREAM.close.assert_called_once()
    assert writer._closed


def test_concurrent_close_calls():
    """Test that concurrent calls to close() only close the stream once."""
    MOCK_STREAM.reset_mock()

    writer = S3Writer(MOCK_STREAM)
    threads = []

    for _ in range(5):
        thread = threading.Thread(target=writer.close)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    MOCK_STREAM.close.assert_called_once()
    assert writer._closed


def test_exit_without_exception():
    """Test __exit__ method when no exception occurs."""
    MOCK_STREAM.reset_mock()

    writer = S3Writer(MOCK_STREAM)
    writer.__exit__(None, None, None)

    MOCK_STREAM.close.assert_called_once()


def test_exit_with_exception(caplog):
    """Test __exit__ method when an exception occurs."""
    MOCK_STREAM.reset_mock()

    writer = S3Writer(MOCK_STREAM)
    test_exception = ValueError("Test exception")
    writer.__exit__(ValueError, test_exception, None)

    # Stream should not be closed on exception
    MOCK_STREAM.close.assert_not_called()
