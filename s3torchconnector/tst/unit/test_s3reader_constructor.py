#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import pytest

from s3torchconnector import S3ReaderConstructor
from s3torchconnector.s3reader import SequentialS3Reader, RangedS3Reader
from s3torchconnector.s3reader.ranged import DEFAULT_BUFFER_SIZE

TEST_BUCKET = "test-bucket"
TEST_KEY = "test-key"


def test_s3readerconstructor_sequential_constructor():
    """Test sequential reader construction"""
    constructor = S3ReaderConstructor.sequential()
    s3reader = constructor(TEST_BUCKET, TEST_KEY, lambda: None, lambda: iter([]))
    assert isinstance(s3reader, SequentialS3Reader)


def test_s3readerconstructor_range_based_constructor():
    """Test range-based reader construction"""
    constructor = S3ReaderConstructor.range_based()
    s3reader = constructor(TEST_BUCKET, TEST_KEY, lambda: None, lambda: iter([]))
    assert isinstance(s3reader, RangedS3Reader)


@pytest.mark.parametrize(
    "buffer_size, expected_buffer_size, expected_enable_buffering",
    [
        (None, DEFAULT_BUFFER_SIZE, True),  # Default buffer size
        (16 * 1024 * 1024, 16 * 1024 * 1024, True),  # Custom buffer size
        (0, 0, False),  # Disabled buffering
    ],
)
def test_s3readerconstructor_range_based_constructor_buffer_configurations(
    buffer_size, expected_buffer_size, expected_enable_buffering
):
    """Test range-based reader construction with different buffer configurations"""
    constructor = S3ReaderConstructor.range_based(buffer_size=buffer_size)
    s3reader = constructor(TEST_BUCKET, TEST_KEY, lambda: None, lambda: iter([]))

    assert isinstance(s3reader, RangedS3Reader)
    assert s3reader._buffer_size == expected_buffer_size
    assert s3reader._enable_buffering is expected_enable_buffering


def test_s3readerconstructor_default_constructor():
    """Test default constructor returns sequential reader"""
    constructor = S3ReaderConstructor.default()
    s3reader = constructor(TEST_BUCKET, TEST_KEY, lambda: None, lambda: iter([]))
    assert isinstance(s3reader, SequentialS3Reader)


def test_s3readerconstructor_get_reader_type_string():
    """Test reader type string generation"""
    assert (
        S3ReaderConstructor.get_reader_type_string(S3ReaderConstructor.sequential())
        == "sequential"
    )
    assert (
        S3ReaderConstructor.get_reader_type_string(S3ReaderConstructor.range_based())
        == "range_based"
    )
    assert S3ReaderConstructor.get_reader_type_string(None) == "sequential"
    assert (
        S3ReaderConstructor.get_reader_type_string(S3ReaderConstructor.default())
        == "sequential"
    )
