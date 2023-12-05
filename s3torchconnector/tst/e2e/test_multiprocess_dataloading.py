#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from __future__ import annotations

import platform
from collections import Counter
from itertools import product
from typing import Tuple, Callable, TYPE_CHECKING

import pytest
import torch
from torch.utils.data import DataLoader, get_worker_info
from torchdata.datapipes.iter import IterableWrapper

from s3torchconnector import S3IterableDataset, S3MapDataset, S3Reader

if TYPE_CHECKING:
    from .conftest import BucketPrefixFixture


start_methods = set(torch.multiprocessing.get_all_start_methods())

if platform.system() == "Darwin":
    # fork and forkserver crash on MacOS, even though it's reported as usable.
    # https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    # https://bugs.python.org/issue?@action=redirect&bpo=33725
    start_methods -= {"fork", "forkserver"}


def from_prefix(cls, image_directory: BucketPrefixFixture, **kwargs):
    return cls.from_prefix(
        s3_uri=f"s3://{image_directory.bucket}/{image_directory.prefix}",
        region=image_directory.region,
        **kwargs,
    )


def from_objects(cls, image_directory: BucketPrefixFixture, **kwargs):
    return cls.from_objects(
        [f"s3://{image_directory.bucket}/{key}" for key in image_directory],
        region=image_directory.region,
        **kwargs,
    )


# Allow us to construct our datasets in tests with either from_prefix or from_objects.
dataset_builders = (from_prefix, from_objects)

test_args = list(product(sorted(start_methods), dataset_builders))


@pytest.mark.parametrize("start_method, dataset_builder", test_args)
def test_s3iterable_dataset_multiprocess_torchdata(
    start_method: str,
    dataset_builder: Callable,
    image_directory: BucketPrefixFixture,
):
    _set_start_method(start_method)
    dataset = dataset_builder(S3IterableDataset, image_directory)

    dataset = IterableWrapper(dataset, deepcopy=False).sharding_filter().map(_read_data)

    batch_size = 2
    num_workers = 3

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

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

    # IterableWrapper has sharding enabled; we'll see each image once.
    assert total_objects == len(image_directory.contents)
    assert uris_seen == {key: 1 for key in image_directory}


@pytest.mark.parametrize("start_method, dataset_builder", test_args)
def test_s3iterable_dataset_multiprocess(
    start_method: str,
    dataset_builder: Callable,
    image_directory: BucketPrefixFixture,
):
    _set_start_method(start_method)
    dataset = dataset_builder(
        S3IterableDataset,
        image_directory,
        transform=_extract_object_data,
    )

    num_workers = 3
    num_epochs = 2
    num_images = len(image_directory.contents)

    dataloader = DataLoader(dataset, num_workers=num_workers)
    counter = 0
    for epoch in range(num_epochs):
        s3keys = Counter()
        worker_count = Counter()
        for ((s3key,), (contents,)), (worker_id, _num_workers) in dataloader:
            s3keys[s3key] += 1
            counter += 1
            worker_count[worker_id.item()] += 1
            assert _num_workers == num_workers
            assert image_directory[s3key] == contents
        assert len(worker_count) == num_workers
        assert all(times_found == num_images for times_found in worker_count.values())
        # Iterable dataset does not do sharding; thus we'll see each image once for each worker.
        assert sum(worker_count.values()) == num_images * num_workers
        assert dict(s3keys) == {key: num_workers for key in image_directory}


@pytest.mark.parametrize("start_method, dataset_builder", test_args)
def test_s3mapdataset_multiprocess(
    start_method: str,
    dataset_builder: Callable,
    image_directory: BucketPrefixFixture,
):
    _set_start_method(start_method)
    dataset = dataset_builder(
        S3MapDataset,
        image_directory,
        transform=_extract_object_data,
    )

    num_workers = 3
    num_epochs = 2
    num_images = len(image_directory.contents)

    dataloader = DataLoader(dataset, num_workers=num_workers)

    for epoch in range(num_epochs):
        s3keys = Counter()
        worker_count = Counter()
        for ((s3key,), (contents,)), (worker_id, _num_workers) in dataloader:
            worker_count[worker_id.item()] += 1
            s3keys[s3key] += 1
            assert _num_workers == num_workers
            assert image_directory[s3key] == contents
        # Map dataset does sharding; we'll see each image once.
        assert sum(worker_count.values()) == num_images
        assert dict(s3keys) == {key: 1 for key in image_directory}
    assert len(dataloader) == num_images


def _set_start_method(start_method: str):
    torch.multiprocessing.set_start_method(start_method, force=True)


def _extract_object_data(s3reader: S3Reader) -> ((str, bytes), (int, int)):
    assert s3reader._stream is None
    return _read_data(s3reader), _get_worker_info()


def _get_worker_info() -> (int, int):
    worker_info = get_worker_info()
    assert worker_info is not None
    return worker_info.id, worker_info.num_workers


def _read_data(s3reader: S3Reader) -> Tuple[str, bytes]:
    return s3reader.key, s3reader.read()
