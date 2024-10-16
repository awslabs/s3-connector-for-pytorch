#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from __future__ import annotations

from collections import Counter
from itertools import product
from typing import Callable, TYPE_CHECKING

import pytest
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

from s3torchconnector import S3IterableDataset, S3MapDataset

if TYPE_CHECKING:
    from .conftest import BucketPrefixFixture, BucketPrefixData


from test_common import _get_start_methods, _read_data, _set_start_method


start_methods = _get_start_methods()

import torch.distributed as dist
def setup(rank, world_size):
    dist.init_process_group(backend="gloo", world_size=world_size, rank=rank, init_method="tcp://127.0.0.1:1234")


def cleanup():
    dist.destroy_process_group()


def from_prefix(
    cls, image_directory: BucketPrefixFixture, rank: int, world_size: int, **kwargs
):
    return cls.from_prefix(
        s3_uri=f"s3://{image_directory.bucket}/{image_directory.prefix}",
        region=image_directory.region,
        rank=rank,
        world_size=world_size,
        transform=_read_data,
        **kwargs,
    )


def from_objects(
    cls, image_directory: BucketPrefixFixture, rank: int, world_size: int, **kwargs
):
    return cls.from_objects(
        [f"s3://{image_directory.bucket}/{key}" for key in image_directory],
        region=image_directory.region,
        rank=rank,
        world_size=world_size,
        transform=_read_data,
        **kwargs,
    )


def dataloader_for_map(dataset_builder, image_directory, num_workers, rank, world_size, batch_size):
    dataset = dataset_builder(S3MapDataset, image_directory, rank, world_size)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
    return dataloader


def dataloader_for_iterable(dataset_builder, image_directory, num_workers, rank, world_size, batch_size):
    dataset = dataset_builder(S3IterableDataset, image_directory, rank, world_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    return dataloader


# Allow us to construct our datasets in tests with either from_prefix or from_objects.
dataset_builders = (from_prefix, from_objects)

# Allow us to construct dataloaders in test with either S3MapDataset or S3IterableDataset
dataloader_builders = (dataloader_for_iterable, dataloader_for_map)

num_workers_to_test = [2]
num_processes_to_test = [1, 2]
test_args = list(
    product(
        sorted(start_methods),
        dataset_builders,
        dataloader_builders,
        num_workers_to_test,
        num_processes_to_test,
    )
)


@pytest.mark.parametrize(
    "start_method, dataset_builder, dataloader_builder, num_workers, num_processes", test_args
)
def test_distributed_training(
    start_method: str,
    dataset_builder: Callable,
    dataloader_builder: Callable,
    num_workers: int,
    num_processes: int,
    image_directory_for_dp: BucketPrefixFixture,
):
    manager = mp.Manager()
    result_queue = manager.Queue()

    mp.start_processes(
        _test_s3iterable_dataset_multiprocess_torchdata,
        args=(
            num_processes,
            num_workers,
            start_method,
            dataset_builder,
            dataloader_builder,
            image_directory_for_dp.get_context_only(),
            result_queue,
        ),
        nprocs=num_processes,
        start_method=start_method,
    )

    # Collect the results from the queue
    results = [result_queue.get() for _ in range(num_processes)]

    # Combine all uris_seen from the results
    combined_uris_seen = Counter()
    for uris_seen in results:
        combined_uris_seen.update(uris_seen)

    # Check if each item in image_directory was seen exactly once
    expected_uris = set(image_directory_for_dp.contents.keys())
    assert set(combined_uris_seen.keys()) == expected_uris
    assert all(count == 1 for count in combined_uris_seen.values())


def _test_s3iterable_dataset_multiprocess_torchdata(
    rank: int,
    world_size: int,
    num_workers: int,
    start_method: str,
    dataset_builder: Callable,
    dataloader_builder: Callable,
    image_directory: BucketPrefixData,
    result_queue: mp.Queue,
):
    setup(rank, world_size)
    _set_start_method(start_method)
    batch_size = 2
    dataloader = dataloader_builder(dataset_builder, image_directory, num_workers, rank, world_size, batch_size)

    total_objects = 0
    uris_seen = Counter()
    for uris, datas in dataloader:
        assert len(uris) == len(datas)
        object_count = len(uris)
        assert object_count <= batch_size
        total_objects += object_count
        for uri, data in zip(uris, datas):
            assert isinstance(uri, str)
            assert isinstance(data, bytes)
            uris_seen[uri] += 1

    cleanup()
    # Put the result in the queue
    result_queue.put(uris_seen)
