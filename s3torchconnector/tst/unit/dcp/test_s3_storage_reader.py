#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from typing import Dict, Any
from unittest.mock import Mock
from hypothesis import given, assume
from hypothesis.strategies import composite, integers, lists

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.planner import LoadPlan, ReadItem, LoadItemType
from torch.distributed.checkpoint.metadata import (
    Metadata,
    MetadataIndex,
    TensorStorageMetadata,
    ChunkStorageMetadata,
)

from s3torchconnector.dcp import S3StorageReader

TEST_REGION = "eu-east-1"
TEST_PATH = "s3://test-bucket/test-checkpoint/"


@composite
def load_plan_with_offsets(draw):
    """Generate LoadPlan with random offsets."""
    offsets = draw(lists(integers(0, 10_000_000), min_size=0, max_size=10_000))

    storage_data = {}
    items = []

    for i, offset in enumerate(offsets):
        metadata_index = MetadataIndex(fqn=f"item{i}", offset=torch.Size([0]), index=0)

        # Mock storage info
        storage_data[metadata_index] = Mock(
            offset=offset,
            length=draw(
                integers(1000, 50000)
            ),  # DCP requires length - use random integers
            relative_path=f"__{draw(integers(0, 7))}_0.distcp",
        )

        items.append(
            Mock(spec=ReadItem, storage_index=metadata_index, type=LoadItemType.TENSOR)
        )

    return LoadPlan(items), storage_data  # type: ignore


@given(load_plan_with_offsets())
def test_s3storage_reader_prepare_local_plan(loadplan_and_storagedata):
    """Test prepare local plan sorts items by storage_data offset."""
    load_plan, storage_data = loadplan_and_storagedata

    s3_storage_reader = S3StorageReader(TEST_REGION, TEST_PATH)
    s3_storage_reader.storage_data = storage_data

    sorted_plan = s3_storage_reader.prepare_local_plan(load_plan)
    offsets = [storage_data[item.storage_index].offset for item in sorted_plan.items]

    # Verify Load Ordering sorts offsets
    assert offsets == sorted(offsets)

    # Verify Load Ordering keeps items the same
    assert len(sorted_plan.items) == len(load_plan.items)
    assert {item.storage_index for item in sorted_plan.items} == {
        item.storage_index for item in load_plan.items
    }


@given(load_plan_with_offsets())
def test_s3storage_reader_dcp_load_uses_load_ordering(loadplan_and_storagedata):
    """Test that DCP automatically calls our load ordering optimization via prepare_local_plan."""
    load_plan, storage_data = loadplan_and_storagedata

    # Skip test cases where input is already sorted
    original_offsets = [
        storage_data[item.storage_index].offset for item in load_plan.items
    ]
    assume(original_offsets != sorted(original_offsets))
    assume(len(original_offsets) > 0)

    # Minimal tensor metadata to satisfy DCP's validation requirements
    state_dict_metadata: Dict[str, Any] = {
        f"item{i}": TensorStorageMetadata(
            properties=Mock(dtype=torch.float32),  # tensor type validation
            size=torch.Size([10]),  # memory allocation
            chunks=[  # chunk info for distributed loading
                ChunkStorageMetadata(offsets=torch.Size([0]), sizes=torch.Size([10]))
            ],
        )
        for i in range(len(load_plan.items))
    }

    # Create S3StorageReader with mock read_metadata (iterable) and read_data
    s3_storage_reader = S3StorageReader(TEST_REGION, TEST_PATH)
    s3_storage_reader.read_metadata = Mock(
        return_value=Metadata(
            state_dict_metadata=state_dict_metadata,  # Real dict for DCP iteration
            storage_data=storage_data,  # Our test data with random offsets
        )
    )
    s3_storage_reader.read_data = Mock()

    # Create state_dict matching the metadata structure
    state_dict = {f"item{i}": torch.zeros(10) for i in range(len(load_plan.items))}

    # 1. In torch/distributed/checkpoint/state_dict_loader.py: dcp.load() calls _load_state_dict;
    # 2. According to torch/distributed/checkpoint/storage.py StorageWriter docstring, _load_state_dict() calls:
    #    read_metadata() > set_up_storage_reader() > prepare_local_plan() > prepare_global_plan() > read_data()
    dcp.load(state_dict, storage_reader=s3_storage_reader)

    # When read_data is called, verify prepare_local_plan was called and sorted the items
    sorted_plan = s3_storage_reader.read_data.call_args[0][0]  # First arg is the plan
    sorted_offsets = [
        storage_data[item.storage_index].offset for item in sorted_plan.items
    ]
    assert sorted_offsets == sorted(sorted_offsets)

    # Verify Load Ordering keeps items the same
    assert len(sorted_plan.items) == len(load_plan.items)
    assert {item.storage_index for item in sorted_plan.items} == {
        item.storage_index for item in load_plan.items
    }
