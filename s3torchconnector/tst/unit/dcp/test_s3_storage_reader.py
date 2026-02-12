#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import os
from unittest.mock import Mock, patch
from hypothesis import given
from hypothesis.strategies import composite, integers, lists

from torch.distributed.checkpoint.planner import LoadPlan, ReadItem

from s3torchconnector.dcp import S3StorageReader
from s3torchconnector.s3reader import S3ReaderConstructor, ItemRange

TEST_REGION = "eu-east-1"
TEST_PATH = "s3://test-bucket/test-checkpoint/"


@composite
def load_plan_with_offsets(draw):
    """Generate LoadPlan with random offsets and lengths."""
    offsets = draw(lists(integers(0, 10_000_000), min_size=1, max_size=10_000))
    lengths = draw(lists(integers(1, 10_000_000), min_size=1, max_size=10_000))

    storage_data = {}
    items = []

    for i, (offset, length) in enumerate(zip(offsets, lengths)):
        storage_index = f"item{i}"
        storage_data[storage_index] = Mock(
            relative_path=f"__{i%8}_0.distcp", offset=offset, length=length
        )
        items.append(Mock(spec=ReadItem, storage_index=storage_index))

    return LoadPlan(items), storage_data


def test_s3storage_reader_prepare_local_plan_empty():
    """Test prepare_local_plan handles empty plans."""
    s3_storage_reader = S3StorageReader(TEST_REGION, TEST_PATH)

    sorted_plan = s3_storage_reader.prepare_local_plan(LoadPlan([]))
    # Output: LoadPlan(items=[], storage_data=None, planner_data=None)

    assert isinstance(sorted_plan, LoadPlan)
    assert len(sorted_plan.items) == 0


@given(load_plan_with_offsets())
def test_s3storage_reader_prepare_local_plan_sorts_items(loadplan_and_storagedata):
    """Test prepare local plan sorts items by storage_data offset."""
    load_plan, storage_data = loadplan_and_storagedata

    s3_storage_reader = S3StorageReader(TEST_REGION, TEST_PATH)
    s3_storage_reader.storage_data = storage_data

    sorted_plan = s3_storage_reader.prepare_local_plan(load_plan)
    sorted_offsets = [
        storage_data[item.storage_index].offset for item in sorted_plan.items
    ]

    # Verify return type
    assert isinstance(sorted_plan, LoadPlan)

    # Verify Load Ordering sorts offsets
    assert sorted_offsets == sorted(sorted_offsets)

    # Verify Load Ordering keeps items the same
    assert len(sorted_plan.items) == len(load_plan.items)
    assert set(sorted_plan.items) == set(load_plan.items)


@given(load_plan_with_offsets())
def test_s3storage_reader_prepare_local_plan_calls_range_injection(
    loadplan_and_storagedata,
):
    """Test prepare_local_plan calls set_item_ranges_by_file() for DCPS3ReaderConstructor."""
    load_plan, storage_data = loadplan_and_storagedata

    constructor = S3ReaderConstructor.dcp_optimized()
    s3_storage_reader = S3StorageReader(
        TEST_REGION, TEST_PATH, reader_constructor=constructor
    )
    s3_storage_reader.storage_data = storage_data

    with patch.object(constructor, "set_item_ranges_by_file") as mock_method:
        s3_storage_reader.prepare_local_plan(load_plan)
        mock_method.assert_called_once_with(load_plan.items, storage_data, TEST_PATH)


@given(load_plan_with_offsets())
def test_s3storage_reader_prepare_local_plan_injects_ranges_correctly(
    loadplan_and_storagedata,
):
    """Test prepare_local_plan correctly injects ranges into DCPS3ReaderConstructor."""
    load_plan, storage_data = loadplan_and_storagedata

    constructor = S3ReaderConstructor.dcp_optimized()
    s3_storage_reader = S3StorageReader(
        TEST_REGION, TEST_PATH, reader_constructor=constructor
    )
    s3_storage_reader.storage_data = storage_data
    s3_storage_reader.prepare_local_plan(load_plan)

    for item in load_plan.items:
        storage_info = storage_data[item.storage_index]
        offset, length = storage_info.offset, storage_info.length
        relative_path = storage_info.relative_path

        expected_range = ItemRange(offset, offset + length)
        s3_uri = f"{TEST_PATH}{relative_path}"
        assert expected_range in constructor._item_ranges_by_file[s3_uri]


@given(load_plan_with_offsets())
def test_s3storage_reader_prepare_local_plan_no_injection_for_other_constructors(
    reader_constructor,
    loadplan_and_storagedata,
):
    """Test prepare_local_plan does NOT inject ranges for non-DCPOptimized reader constructors."""
    load_plan, storage_data = loadplan_and_storagedata

    s3_storage_reader = S3StorageReader(
        TEST_REGION, TEST_PATH, reader_constructor=reader_constructor
    )
    s3_storage_reader.storage_data = storage_data

    result = s3_storage_reader.prepare_local_plan(load_plan)
    assert len(result.items) == len(load_plan.items)

    # Verify no injection occurred - regular constructors don't have _item_ranges_by_file
    assert not hasattr(reader_constructor, "_item_ranges_by_file")
