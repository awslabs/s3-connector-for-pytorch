#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from unittest.mock import Mock
from hypothesis import given
from hypothesis.strategies import composite, integers, lists

from torch.distributed.checkpoint.planner import LoadPlan, ReadItem

from s3torchconnector.dcp import S3StorageReader

TEST_REGION = "eu-east-1"
TEST_PATH = "s3://test-bucket/test-checkpoint/"


@composite
def load_plan_with_offsets(draw):
    """Generate LoadPlan with random offsets."""
    offsets = draw(lists(integers(0, 10_000_000), min_size=1, max_size=10_000))

    storage_data = {}
    items = []

    for i, offset in enumerate(offsets):
        storage_index = f"item{i}"
        storage_data[storage_index] = Mock(offset=offset)
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
def test_s3storage_reader_prepare_local_plan(loadplan_and_storagedata):
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
