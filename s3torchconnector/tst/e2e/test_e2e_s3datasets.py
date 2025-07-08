#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import pickle

import pytest
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.datapipes.datapipe import MapDataPipe
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe

from s3torchconnector import (
    S3IterableDataset,
    S3MapDataset,
    S3ClientConfig,
    S3ReaderConstructor,
)
from s3torchconnector.s3reader import S3ReaderConstructorProtocol


def test_s3iterable_dataset_images_10_from_prefix(
    image_directory, reader_constructor: S3ReaderConstructorProtocol
):
    s3_uri = f"s3://{image_directory.bucket}/{image_directory.prefix}"
    dataset = S3IterableDataset.from_prefix(
        s3_uri=s3_uri,
        region=image_directory.region,
        reader_constructor=reader_constructor,
    )
    assert isinstance(dataset, S3IterableDataset)
    _verify_image_iterable_dataset(image_directory, dataset)


def test_s3mapdataset_images_10_from_prefix(
    image_directory, reader_constructor: S3ReaderConstructorProtocol
):
    s3_uri = f"s3://{image_directory.bucket}/{image_directory.prefix}"
    dataset = S3MapDataset.from_prefix(
        s3_uri=s3_uri,
        region=image_directory.region,
        reader_constructor=reader_constructor,
    )
    assert isinstance(dataset, S3MapDataset)
    assert len(dataset) == 10

    # Intentional usage to emphasize the accessor dataset[index] works.
    for index in range(len(dataset)):
        key = dataset[index].key
        if _expect_sorted_list_results(image_directory.storage_class):
            # S3 Express does not guarantee a sorted order for listed keys.
            assert key == f"{image_directory.prefix}img{index:03d}.jpg"
        assert dataset[index].read() == image_directory[key]


@pytest.mark.parametrize(
    "batch_size, expected_batch_count",
    [(1, 10), (2, 5), (4, 3), (10, 1)],
)
def test_dataloader_10_images_s3iterable_dataset(
    batch_size: int,
    expected_batch_count: int,
    image_directory,
    reader_constructor: S3ReaderConstructorProtocol,
):
    local_dataloader = _create_local_dataloader(image_directory, batch_size)
    assert isinstance(local_dataloader.dataset, IterDataPipe)

    s3_uri = f"s3://{image_directory.bucket}/{image_directory.prefix}"
    s3_dataset = S3IterableDataset.from_prefix(
        s3_uri=s3_uri,
        region=image_directory.region,
        transform=lambda obj: obj.read(),
        reader_constructor=reader_constructor,
    )

    s3_dataloader = _pytorch_dataloader(s3_dataset, batch_size)
    assert s3_dataloader is not None
    assert isinstance(s3_dataloader.dataset, S3IterableDataset)

    _compare_dataloaders(local_dataloader, s3_dataloader, expected_batch_count)


@pytest.mark.parametrize(
    "batch_size, expected_batch_count",
    [(1, 10), (2, 5), (4, 3), (10, 1)],
)
def test_dataloader_10_images_s3mapdataset(
    batch_size: int,
    expected_batch_count: int,
    image_directory,
    reader_constructor: S3ReaderConstructorProtocol,
):
    local_dataloader = _create_local_dataloader(image_directory, batch_size, True)
    assert isinstance(local_dataloader.dataset, MapDataPipe)

    s3_uri = f"s3://{image_directory.bucket}/{image_directory.prefix}"
    s3_dataset = S3MapDataset.from_prefix(
        s3_uri=s3_uri,
        region=image_directory.region,
        transform=lambda obj: obj.read(),
        reader_constructor=reader_constructor,
    )
    s3_dataloader = _pytorch_dataloader(s3_dataset, batch_size)
    assert s3_dataloader is not None
    assert isinstance(s3_dataloader.dataset, S3MapDataset)

    _compare_dataloaders(local_dataloader, s3_dataloader, expected_batch_count)


def test_dataset_unpickled_iterates(
    image_directory, reader_constructor: S3ReaderConstructorProtocol
):
    s3_uri = f"s3://{image_directory.bucket}/{image_directory.prefix}"
    dataset = S3IterableDataset.from_prefix(
        s3_uri=s3_uri,
        region=image_directory.region,
        reader_constructor=reader_constructor,
    )
    assert isinstance(dataset, S3IterableDataset)
    unpickled = pickle.loads(pickle.dumps(dataset))

    expected = [i.key for i in dataset]
    actual = [i.key for i in unpickled]

    assert expected == actual


def test_unsigned_client(reader_constructor: S3ReaderConstructorProtocol):
    s3_uri = "s3://s3torchconnector-demo/geonet/images/"
    region = "us-east-1"
    s3_dataset = S3MapDataset.from_prefix(
        s3_uri=s3_uri,
        region=region,
        transform=lambda obj: obj.read(),
        s3client_config=S3ClientConfig(unsigned=True),
        reader_constructor=reader_constructor,
    )
    s3_dataloader = _pytorch_dataloader(s3_dataset)
    assert s3_dataloader is not None
    assert isinstance(s3_dataloader.dataset, S3MapDataset)
    assert len(s3_dataloader) >= 1296


@pytest.mark.parametrize(
    "reader_a,reader_b",
    [
        (S3ReaderConstructor.sequential(), S3ReaderConstructor.range_based()),
        (S3ReaderConstructor.sequential(), S3ReaderConstructor.range_based(0)),
        (S3ReaderConstructor.range_based(), S3ReaderConstructor.range_based(0)),
    ],
)
def test_all_reader_combinations_produce_identical_results(
    image_directory, reader_a, reader_b
):
    """Test that all reader type combinations produce identical results with real S3 data"""
    s3_uri = f"s3://{image_directory.bucket}/{image_directory.prefix}"

    datasets = [
        S3MapDataset.from_prefix(
            s3_uri=s3_uri,
            region=image_directory.region,
            reader_constructor=reader,
            transform=lambda obj: obj.read(),
        )
        for reader in [reader_a, reader_b]
    ]

    dataloaders = [_pytorch_dataloader(ds, batch_size=2) for ds in datasets]
    expected_batch_count = (len(datasets[0]) + 1) // 2

    _compare_dataloaders(*dataloaders, expected_batch_count)


def test_s3mapdataset_user_agent(
    image_directory, reader_constructor: S3ReaderConstructorProtocol
):
    """Test that user agent includes correct dataset type and reader type for S3MapDataset."""
    map_dataset = S3MapDataset.from_prefix(
        s3_uri=f"s3://{image_directory.bucket}/{image_directory.prefix}",
        region=image_directory.region,
        reader_constructor=reader_constructor,
    )

    # Trigger client initialization
    first_item = map_dataset[0]
    first_item.read(1)

    user_agent = map_dataset._client.user_agent_prefix
    reader_type_string = S3ReaderConstructor.get_reader_type_string(reader_constructor)
    assert "md/dataset#map" in user_agent
    assert f"md/reader_type#{reader_type_string}" in user_agent


def test_s3iterabledataset_user_agent(
    image_directory, reader_constructor: S3ReaderConstructorProtocol
):
    """Test that user agent includes correct dataset type and reader type for S3IterableDataset."""
    iter_dataset = S3IterableDataset.from_prefix(
        s3_uri=f"s3://{image_directory.bucket}/{image_directory.prefix}",
        region=image_directory.region,
        reader_constructor=reader_constructor,
    )

    # Trigger client initialization
    first_item = next(iter(iter_dataset))
    first_item.read(1)

    user_agent = iter_dataset._client.user_agent_prefix
    reader_type_string = S3ReaderConstructor.get_reader_type_string(reader_constructor)
    assert "md/dataset#iterable" in user_agent
    assert f"md/reader_type#{reader_type_string}" in user_agent


def _compare_dataloaders(
    local_dataloader: DataLoader, s3_dataloader: DataLoader, expected_batch_count: int
):
    assert s3_dataloader.batch_size == local_dataloader.batch_size

    batch_count = _get_dataloader_len(s3_dataloader)
    assert batch_count == expected_batch_count

    # Iterable datasets aren't required to be ordered, so we check equality of the entire dataset
    # rather than checking each batch is equal.
    local_objs = set()
    s3_objs = set()
    for local_batch, s3_batch in zip(local_dataloader, s3_dataloader):
        # Assert batch samples are of equal lengths. We are not asserting equality with batch_size
        # due to possible remainder from division for the last batch.
        assert len(local_batch) == len(s3_batch)
        for local_item in local_batch:
            local_objs.add(local_item)
        for s3_item in s3_batch:
            s3_objs.add(s3_item)
    assert local_objs == s3_objs

    # TODO: Calling len before zip causes `TypeError: cannot pickle '_io.BufferedReader' object`
    local_batch_count = _get_dataloader_len(local_dataloader)
    assert local_batch_count == expected_batch_count


def _verify_image_iterable_dataset(
    image_directory,
    dataset: S3IterableDataset,
):
    assert dataset is not None
    for index, fileobj in enumerate(dataset):
        assert fileobj is not None
        if _expect_sorted_list_results(image_directory.storage_class):
            # S3 Express does not guarantee a sorted order for listed keys.
            assert fileobj.key == f"{image_directory.prefix}img{index:03d}.jpg"
        assert image_directory[fileobj.key] == fileobj.read()


def _pytorch_dataloader(
    dataset: Dataset, batch_size: int = 1
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=0, shuffle=False
    )


def _create_local_dataloader(
    image_directory, batch_size: int = 1, with_map_dataset: bool = False
):
    dataset = IterableWrapper([image_directory[key] for key in image_directory])
    if with_map_dataset:
        wrapper = IterableWrapper(enumerate(dataset))
        dataset = wrapper.to_map_datapipe()
    return _pytorch_dataloader(dataset, batch_size)


def _get_dataloader_len(dataloader: DataLoader):
    if isinstance(dataloader.dataset, MapDataPipe) or isinstance(
        dataloader.dataset, S3MapDataset
    ):
        return len(dataloader)
    else:
        return len(list(dataloader))


def _expect_sorted_list_results(storage_class: str) -> bool:
    return storage_class != "EXPRESS_ONEZONE"
