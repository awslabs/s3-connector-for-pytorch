#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import pytest
import sys
from unittest.mock import Mock

from s3torchconnector import S3ReaderConstructor
from s3torchconnector.s3reader import (
    SequentialS3Reader,
    RangedS3Reader,
    DCPOptimizedS3Reader,
    DCPOptimizedConstructor,
)
from s3torchconnector.s3reader.dcp_optimized import ItemRange, DEFAULT_MAX_GAP_SIZE
from s3torchconnector.s3reader.ranged import DEFAULT_BUFFER_SIZE

from torch.distributed.checkpoint.planner import ReadItem
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed.checkpoint.filesystem import _StorageInfo

from .test_s3reader_common import TEST_BUCKET, TEST_KEY, MOCK_OBJECT_INFO, MOCK_STREAM

# ---------- basic constructor tests -----------


def test_s3readerconstructor_default_constructor():
    """Test default constructor returns sequential reader"""
    constructor = S3ReaderConstructor.default()
    s3reader = constructor(TEST_BUCKET, TEST_KEY, MOCK_OBJECT_INFO, MOCK_STREAM)
    assert isinstance(s3reader, SequentialS3Reader)


def test_s3readerconstructor_sequential_constructor():
    """Test sequential reader construction"""
    constructor = S3ReaderConstructor.sequential()
    s3reader = constructor(TEST_BUCKET, TEST_KEY, MOCK_OBJECT_INFO, MOCK_STREAM)
    assert isinstance(s3reader, SequentialS3Reader)


def test_s3readerconstructor_range_based_constructor():
    """Test range-based reader construction"""
    constructor = S3ReaderConstructor.range_based()
    s3reader = constructor(TEST_BUCKET, TEST_KEY, MOCK_OBJECT_INFO, MOCK_STREAM)
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
    s3reader = constructor(TEST_BUCKET, TEST_KEY, MOCK_OBJECT_INFO, MOCK_STREAM)

    assert isinstance(s3reader, RangedS3Reader)
    assert s3reader._buffer_size == expected_buffer_size
    assert s3reader._enable_buffering is expected_enable_buffering


# ---------- dcp_optimized constructor tests ----------


def test_s3readerconstructor_dcp_optimized_constructor():
    """Test DCP optimized reader construction"""
    constructor = S3ReaderConstructor.dcp_optimized()
    assert isinstance(constructor, DCPOptimizedConstructor)

    # Without ranges, should fallback to SequentialS3Reader
    s3reader = constructor(TEST_BUCKET, TEST_KEY, MOCK_OBJECT_INFO, MOCK_STREAM)
    assert isinstance(s3reader, SequentialS3Reader)

    # With ranges, should create DCPOptimizedS3Reader
    constructor._item_ranges_by_file = {TEST_KEY: [ItemRange(0, 100)]}
    s3reader = constructor(TEST_BUCKET, TEST_KEY, MOCK_OBJECT_INFO, MOCK_STREAM)
    assert isinstance(s3reader, DCPOptimizedS3Reader)


# * max_gap_size tests


def test_dcp_optimized_constructor_default_max_gap_size():
    """Test max_gap_size parameter defaults and propagation"""

    constructor = S3ReaderConstructor.dcp_optimized()
    assert isinstance(constructor, DCPOptimizedConstructor)
    assert constructor._max_gap_size == DEFAULT_MAX_GAP_SIZE

    constructor._item_ranges_by_file = {TEST_KEY: [ItemRange(0, 100)]}
    s3reader = constructor(TEST_BUCKET, TEST_KEY, MOCK_OBJECT_INFO, MOCK_STREAM)
    assert isinstance(s3reader, DCPOptimizedS3Reader)
    assert s3reader._max_gap_size == DEFAULT_MAX_GAP_SIZE


@pytest.mark.parametrize("max_gap_size", [0, 8 * 1024 * 1024, 1024 * 1024 * 1024])
def test_dcp_optimized_constructor_custom_max_gap_size(max_gap_size):
    """Test max_gap_size parameter defaults and propagation"""

    constructor = S3ReaderConstructor.dcp_optimized(max_gap_size=max_gap_size)
    assert isinstance(constructor, DCPOptimizedConstructor)
    assert constructor._max_gap_size == max_gap_size

    constructor._item_ranges_by_file = {TEST_KEY: [ItemRange(0, 100)]}
    s3reader = constructor(TEST_BUCKET, TEST_KEY, MOCK_OBJECT_INFO, MOCK_STREAM)
    assert isinstance(s3reader, DCPOptimizedS3Reader)
    assert s3reader._max_gap_size == max_gap_size


@pytest.mark.parametrize("max_gap_size", [sys.maxsize, float("inf"), 0.5])
def test_dcp_optimized_constructor_max_gap_size_edge_cases(max_gap_size):
    """Test max_gap_size parameter defaults and propagation"""

    constructor = S3ReaderConstructor.dcp_optimized(max_gap_size=max_gap_size)
    assert isinstance(constructor, DCPOptimizedConstructor)
    assert constructor._max_gap_size == max_gap_size

    constructor._item_ranges_by_file = {TEST_KEY: [ItemRange(0, 100)]}
    s3reader = constructor(TEST_BUCKET, TEST_KEY, MOCK_OBJECT_INFO, MOCK_STREAM)
    assert isinstance(s3reader, DCPOptimizedS3Reader)
    assert s3reader._max_gap_size == max_gap_size


@pytest.mark.parametrize(
    "max_gap_size,expected_error",
    [
        (-1, ValueError),
        ("1", TypeError),
        ([1], TypeError),
        (None, TypeError),
    ],
)
def test_dcp_optimized_constructor_invalid_max_gap_size(max_gap_size, expected_error):
    """Test parameter validation for max_gap_size"""
    with pytest.raises(expected_error):
        S3ReaderConstructor.dcp_optimized(max_gap_size)


# * set_item_ranges_by_file()


def test_dcp_optimized_constructor_set_item_ranges_by_file_empty_file():
    """Test empty plan is allowed (disallowed during call)"""
    constructor = S3ReaderConstructor.dcp_optimized()

    constructor.set_item_ranges_by_file([], {})
    assert constructor._item_ranges_by_file == {}


@pytest.mark.parametrize(
    "relative_path",
    [
        ("__0_0.distcp"),
        ("nested/path/to/file/__0_0.distcp"),
        ("prefix_strategy/shard1/epoch_5/__0_0.distcp"),
    ],
)
def test_dcp_optimized_constructor_set_item_ranges_by_file_filename_extraction(
    relative_path,
):
    """Test filename extraction from various path formats"""
    constructor = S3ReaderConstructor.dcp_optimized()

    metadata_index = MetadataIndex("idx")
    read_item = Mock(spec=ReadItem, storage_index=metadata_index)
    storage_data = {
        metadata_index: _StorageInfo(relative_path=relative_path, offset=0, length=100)
    }

    constructor.set_item_ranges_by_file([read_item], storage_data)
    assert relative_path in constructor._item_ranges_by_file


def test_dcp_optimized_constructor_set_item_ranges_by_file_multiple_items():
    """Test set_item_ranges_by_file with different ReadItems"""
    constructor = S3ReaderConstructor.dcp_optimized()

    metadata_indices = [MetadataIndex(f"idx{i}") for i in range(3)]
    read_items = [
        Mock(spec=ReadItem, storage_index=metadata_indices[i]) for i in range(3)
    ]
    storage_data = {
        metadata_indices[0]: _StorageInfo(
            relative_path="file1.distcp", offset=0, length=100
        ),
        metadata_indices[1]: _StorageInfo(
            relative_path="file1.distcp", offset=100, length=50
        ),
        metadata_indices[2]: _StorageInfo(
            relative_path="file2.distcp", offset=0, length=200
        ),
    }

    constructor.set_item_ranges_by_file(read_items, storage_data)  # type: ignore

    assert "file1.distcp" in constructor._item_ranges_by_file
    assert "file2.distcp" in constructor._item_ranges_by_file
    assert len(constructor._item_ranges_by_file["file1.distcp"]) == 2
    assert len(constructor._item_ranges_by_file["file2.distcp"]) == 1
    assert constructor._item_ranges_by_file["file1.distcp"][0] == ItemRange(0, 100)
    assert constructor._item_ranges_by_file["file1.distcp"][1] == ItemRange(100, 150)


def test_dcp_optimized_constructor_set_item_ranges_by_file_multiple_calls():
    """Test constructor handles multiple calls to set_item_ranges_by_file"""
    constructor = S3ReaderConstructor.dcp_optimized()

    # First call
    metadata_index1 = MetadataIndex("idx1")
    read_item1 = Mock(spec=ReadItem, storage_index=metadata_index1)
    storage_data1 = {
        metadata_index1: _StorageInfo(
            relative_path="file1.distcp", offset=0, length=100
        )
    }
    constructor.set_item_ranges_by_file([read_item1], storage_data1)

    # Second call should replace previous ranges
    metadata_index2 = MetadataIndex("idx2")
    read_item2 = Mock(spec=ReadItem, storage_index=metadata_index2)
    storage_data2 = {
        metadata_index2: _StorageInfo(
            relative_path="file2.distcp", offset=0, length=200
        )
    }
    constructor.set_item_ranges_by_file([read_item2], storage_data2)

    # Only second call's data should remain
    assert "file1.distcp" not in constructor._item_ranges_by_file
    assert "file2.distcp" in constructor._item_ranges_by_file
    assert len(constructor._item_ranges_by_file["file2.distcp"]) == 1
    assert constructor._item_ranges_by_file["file2.distcp"][0] == ItemRange(0, 200)


# * __call__ tests


def test_dcp_optimized_constructor_call_with_invalid_ranges():
    """Test dcp_optimized constructor __call__ falls back to SequentialS3Reader when no item ranges for the file"""
    constructor = S3ReaderConstructor.dcp_optimized()
    assert isinstance(constructor, DCPOptimizedConstructor)

    # Test with no ranges - should fallback
    constructor._item_ranges_by_file = {}
    s3reader = constructor(TEST_BUCKET, TEST_KEY, MOCK_OBJECT_INFO, MOCK_STREAM)
    assert isinstance(s3reader, SequentialS3Reader)

    # Test with ranges for different file - should fallback
    constructor._item_ranges_by_file = {"not_test_key.distcp": [ItemRange(0, 100)]}
    s3reader = constructor(TEST_BUCKET, TEST_KEY, MOCK_OBJECT_INFO, MOCK_STREAM)
    assert isinstance(s3reader, SequentialS3Reader)

    # Test with empty range - should raise error
    constructor._item_ranges_by_file = {TEST_KEY: []}
    with pytest.raises(ValueError):
        s3reader = constructor(TEST_BUCKET, TEST_KEY, MOCK_OBJECT_INFO, MOCK_STREAM)


def test_dcp_optimized_constructor_call_relative_path_matching():
    """Test __call__ method with relative path matching"""
    constructor = S3ReaderConstructor.dcp_optimized()
    constructor._item_ranges_by_file = {
        "shard1/epoch_5/__0_0.distcp": [ItemRange(0, 100)]
    }

    # Should match when key ends with relative_path
    key = "base/shard1/epoch_5/__0_0.distcp"
    s3reader = constructor(TEST_BUCKET, key, MOCK_OBJECT_INFO, MOCK_STREAM)
    assert isinstance(s3reader, DCPOptimizedS3Reader)
    assert s3reader.key == key

    key = "shard1/epoch_5/__0_0.distcp"
    # Should also match shorter key that ends with relative_path
    s3reader = constructor(TEST_BUCKET, key, MOCK_OBJECT_INFO, MOCK_STREAM)
    assert isinstance(s3reader, DCPOptimizedS3Reader)
    assert s3reader.key == key

    # No match - should fallback to SequentialS3Reader
    key = "different/path/__1_0.distcp"
    s3reader = constructor(TEST_BUCKET, key, MOCK_OBJECT_INFO, MOCK_STREAM)
    assert isinstance(s3reader, SequentialS3Reader)
    assert s3reader.key == key

    # Same filename, different path - should fallback to SequentialS3Reader
    key = "same/filename/different/path/__0_0.distcp"
    s3reader = constructor(TEST_BUCKET, key, MOCK_OBJECT_INFO, MOCK_STREAM)
    assert isinstance(s3reader, SequentialS3Reader)
    assert s3reader.key == key


# ----------


@pytest.mark.parametrize(
    "constructor, expected_type",
    [
        (S3ReaderConstructor.sequential(), "sequential"),
        (S3ReaderConstructor.range_based(), "range_based"),
        (S3ReaderConstructor.dcp_optimized(), "dcp_optimized"),
        (None, "sequential"),
        (S3ReaderConstructor.default(), "sequential"),
    ],
)
def test_s3readerconstructor_get_reader_type_string(constructor, expected_type):
    """Test reader type string generation"""
    assert S3ReaderConstructor.get_reader_type_string(constructor) == expected_type
