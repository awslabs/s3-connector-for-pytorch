#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from __future__ import annotations

import platform
from collections import Counter
from itertools import product
from typing import Callable, TYPE_CHECKING
import hashlib
from math import prod

import pytest
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

from s3torchconnector import S3IterableDataset, S3MapDataset, S3ReaderConstructor
from s3torchconnector.s3reader import S3ReaderConstructorProtocol

if TYPE_CHECKING:
    from .conftest import BucketPrefixFixture, BucketPrefixData


from test_common import _get_fork_methods, _read_data, _set_start_method

start_methods = _get_fork_methods()

import torch.distributed as dist


def setup(unique_port, rank, world_size):
    dist.init_process_group(
        backend="gloo",
        world_size=world_size,
        rank=rank,
        init_method=f"tcp://127.0.0.1:{unique_port}",
    )


def cleanup():
    dist.destroy_process_group()


def from_prefix(
    cls,
    image_directory: BucketPrefixFixture,
    reader_constructor: S3ReaderConstructorProtocol,
    **kwargs,
):
    return cls.from_prefix(
        s3_uri=f"s3://{image_directory.bucket}/{image_directory.prefix}",
        region=image_directory.region,
        transform=_read_data,
        reader_constructor=reader_constructor,
        **kwargs,
    )


def from_objects(
    cls,
    image_directory: BucketPrefixFixture,
    reader_constructor: S3ReaderConstructorProtocol,
    **kwargs,
):
    return cls.from_objects(
        [f"s3://{image_directory.bucket}/{key}" for key in image_directory],
        region=image_directory.region,
        transform=_read_data,
        reader_constructor=reader_constructor,
        **kwargs,
    )


def dataloader_for_map(
    dataset_builder, image_directory, num_workers, batch_size, reader_constructor
):
    dataset = dataset_builder(
        S3MapDataset, image_directory, reader_constructor=reader_constructor
    )
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        multiprocessing_context=mp.get_context(),
    )
    return dataloader


def dataloader_for_iterable(
    dataset_builder, image_directory, num_workers, batch_size, reader_constructor
):
    dataset = dataset_builder(
        cls=S3IterableDataset,
        image_directory=image_directory,
        enable_sharding=True,
        reader_constructor=reader_constructor,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        multiprocessing_context=mp.get_context(),
    )
    return dataloader


# Allow us to construct our datasets in tests with either from_prefix or from_objects.
dataset_builders = [from_prefix, from_objects]

# Allow us to construct dataloaders in test with either S3MapDataset or S3IterableDataset
dataloader_builders = [dataloader_for_iterable, dataloader_for_map]

# Allow us to construct our datasets in tests with either both reader types.
reader_constructors = [
    S3ReaderConstructor.sequential(),  # Sequential Reader
    S3ReaderConstructor.range_based(),  # Default range-based reader, with buffer
    S3ReaderConstructor.range_based(buffer_size=0),  # range-based reader, no buffer
]
num_workers_to_test = [1, 3]
num_processes_to_test = [1, 3]
test_args = list(
    product(
        sorted(start_methods),
        dataset_builders,
        dataloader_builders,
        num_workers_to_test,
        num_processes_to_test,
        reader_constructors,
    )
)


@pytest.mark.skipif(
    platform.system() == "Darwin" and platform.machine() == "x86_64",
    reason="Temporarily disabled for macOS x86_64 due to long distributed training test execution times for macos-15 leading to credential timeout.",
)
@pytest.mark.parametrize(
    "start_method, dataset_builder, dataloader_builder, num_workers, num_processes, reader_constructor",
    test_args,
)
def test_distributed_training(
    request,
    start_method: str,
    dataset_builder: Callable,
    dataloader_builder: Callable,
    num_workers: int,
    num_processes: int,
    image_directory_for_dp: BucketPrefixFixture,
    reader_constructor: S3ReaderConstructorProtocol,
):
    """Calculate a unique port number based on the input parameters
    This ensures that each test case runs on a different port
    to avoid conflicts when running in parallel
    """
    parameters = [
        (start_methods, start_method),
        (dataset_builders, dataset_builder),
        (dataloader_builders, dataloader_builder),
        (num_workers_to_test, num_workers),
        (num_processes_to_test, num_processes),
        (reader_constructors, reader_constructor),
    ]
    options, values = zip(*parameters)
    multipliers = [prod(map(len, options[i + 1 :])) for i in range(len(options))]
    port_offset = sum(o.index(v) * m for o, v, m in zip(options, values, multipliers))
    unique_port = 2000 + port_offset

    reader_type_string = S3ReaderConstructor.get_reader_type_string(reader_constructor)

    print(
        f"Testing {request.node.name} with start_method={start_method}, "
        f"dataset_builder={dataset_builder.__name__}, dataloader_builder={dataloader_builder.__name__}, "
        f"num_workers={num_workers}, num_processes={num_processes}, reader_type_string={reader_type_string}, "
        f"unique_port={unique_port}"
    )

    manager = mp.Manager()
    result_queue = manager.Queue()

    mp.start_processes(
        _test_s3iterable_dataset_multiprocess_torchdata,
        args=(
            unique_port,
            num_processes,
            num_workers,
            start_method,
            dataset_builder,
            dataloader_builder,
            image_directory_for_dp.get_data_snapshot(),
            result_queue,
            reader_constructor,
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

    # Check all items in image_directory were seen
    expected_uris = set(image_directory_for_dp.contents.keys())
    assert set(combined_uris_seen.keys()) == expected_uris

    """When conducting distributed training tests, be cautious about the number of files (images) in the test dataset.
    If the total number of images cannot be evenly divided by the number of workers,
    the DistributedSampler will duplicate a subset of the images across workers to ensure an equal
    distribution of data among all processes. This duplication of images will cause
    integration distributed training test to fail.
    """
    assert all(count == 1 for count in combined_uris_seen.values())


def _test_s3iterable_dataset_multiprocess_torchdata(
    rank: int,
    unique_port: int,
    world_size: int,
    num_workers: int,
    start_method: str,
    dataset_builder: Callable,
    dataloader_builder: Callable,
    image_directory: BucketPrefixData,
    result_queue: mp.Queue,
    reader_constructor: S3ReaderConstructorProtocol,
):
    setup(unique_port, rank, world_size)
    _set_start_method(start_method)
    batch_size = 2
    dataloader = dataloader_builder(
        dataset_builder, image_directory, num_workers, batch_size, reader_constructor
    )

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
