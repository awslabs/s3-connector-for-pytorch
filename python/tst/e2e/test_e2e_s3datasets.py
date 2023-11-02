import os
from typing import Tuple

import pytest
import torch
import torchdata
import torchvision
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.datapipes.datapipe import MapDataPipe
from torch.utils.data.datapipes.utils.common import StreamWrapper
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe

from s3dataset._s3dataset import MountpointS3Client
from s3dataset.s3object import S3Object
from s3dataset.s3iterable_dataset import S3IterableDataset
from s3dataset.s3mapstyle_dataset import S3MapStyleDataset


E2E_TEST_BUCKET = "dataset-it-bucket"
E2E_BUCKET_PREFIX = "e2e-tests/images-10/img"
E2E_TEST_REGION = "eu-west-2"
LOCAL_DATASET_RELATIVE_PATH = "../resources/images-10/"
ABSOLUTE_PATH = os.path.dirname(__file__)


def test_s3iterable_dataset_images_10_with_client():
    client = MountpointS3Client(E2E_TEST_REGION)
    dataset = S3IterableDataset.from_bucket(
        E2E_TEST_BUCKET, prefix=E2E_BUCKET_PREFIX, client=client
    )
    _verify_image_iterable_dataset(LOCAL_DATASET_RELATIVE_PATH, dataset)


def test_s3iterable_dataset_images_10_with_region():
    dataset = S3IterableDataset.from_bucket(
        E2E_TEST_BUCKET, prefix=E2E_BUCKET_PREFIX, region=E2E_TEST_REGION
    )
    _verify_image_iterable_dataset(LOCAL_DATASET_RELATIVE_PATH, dataset)


def test_s3mapstyle_dataset_images_10_with_client():
    client = MountpointS3Client(E2E_TEST_REGION)
    dataset = S3MapStyleDataset.from_bucket(
        E2E_TEST_BUCKET, prefix=E2E_BUCKET_PREFIX, client=client
    )
    assert isinstance(dataset, S3MapStyleDataset)
    assert len(dataset) == 10
    # Intentional usage to emphasize the accessor dataset[index] works.
    for index in range(len(dataset)):
        _compare_dataset_img_against_local(
            f"{LOCAL_DATASET_RELATIVE_PATH}img{index}.jpg", dataset[index]
        )


def test_s3mapstyle_dataset_images_10_with_region():
    dataset = S3MapStyleDataset.from_bucket(
        E2E_TEST_BUCKET, prefix=E2E_BUCKET_PREFIX, region=E2E_TEST_REGION
    )
    assert isinstance(dataset, S3MapStyleDataset)
    assert len(dataset) == 10
    # Intentional usage to emphasize the accessor dataset[index] works.
    for index in range(len(dataset)):
        _compare_dataset_img_against_local(
            f"{LOCAL_DATASET_RELATIVE_PATH}img{index}.jpg", dataset[index]
        )


@pytest.mark.parametrize(
    "batch_size, expected_batch_count",
    [(1, 10), (2, 5), (3, 4), (4, 3), (5, 2), (10, 1)],
)
def test_dataloader_10_images_s3iterable_dataset(
    batch_size: int, expected_batch_count: int
):
    local_dataloader = _create_local_dataloader(batch_size)
    assert isinstance(local_dataloader.dataset, IterDataPipe)

    s3_dataset = S3IterableDataset.from_bucket(
        E2E_TEST_BUCKET,
        prefix=E2E_BUCKET_PREFIX,
        region=E2E_TEST_REGION,
        transform=lambda obj: _map_image_to_tensor(Image.open(obj)),
    )
    s3_dataloader = _pytorch_dataloader(s3_dataset, batch_size)
    assert s3_dataloader is not None
    assert isinstance(s3_dataloader.dataset, S3IterableDataset)

    _compare_dataloaders(local_dataloader, s3_dataloader, expected_batch_count)


@pytest.mark.parametrize(
    "batch_size, expected_batch_count",
    [(1, 10), (2, 5), (3, 4), (4, 3), (5, 2), (10, 1)],
)
def test_dataloader_10_images_s3mapstyle_dataset(
    batch_size: int, expected_batch_count: int
):
    local_dataloader = _create_local_dataloader(batch_size, True)
    assert isinstance(local_dataloader.dataset, MapDataPipe)

    s3_dataset = S3MapStyleDataset.from_bucket(
        E2E_TEST_BUCKET,
        prefix=E2E_BUCKET_PREFIX,
        region=E2E_TEST_REGION,
        transform=lambda obj: _map_image_to_tensor(Image.open(obj)),
    )
    s3_dataloader = _pytorch_dataloader(s3_dataset, batch_size)
    assert s3_dataloader is not None
    assert isinstance(s3_dataloader.dataset, S3MapStyleDataset)

    _compare_dataloaders(local_dataloader, s3_dataloader, expected_batch_count)


def _compare_dataloaders(
    local_dataloader: DataLoader, s3_dataloader: DataLoader, expected_batch_count: int
):
    assert s3_dataloader.batch_size == local_dataloader.batch_size

    batch_count = _get_dataloader_len(s3_dataloader)
    assert batch_count == expected_batch_count

    for local_batch, s3_batch in zip(local_dataloader, s3_dataloader):
        # Assert batch samples are of equal lengths.
        # We are not asserting equality with batch_size due to possible
        # remainder from division for the last batch.
        assert len(local_batch) == len(s3_batch)
        for local_item, s3_item in zip(local_batch, s3_batch):
            # Assert tensors are equal
            assert torch.equal(local_item, s3_item)

    # TODO: Calling len before zip causes `TypeError: cannot pickle '_io.BufferedReader' object`
    local_batch_count = _get_dataloader_len(local_dataloader)
    assert local_batch_count == expected_batch_count


def _load_image(img_file_relative_path: str) -> Image:
    full_path = os.path.join(ABSOLUTE_PATH, img_file_relative_path)
    return Image.open(full_path)


def _compare_dataset_img_against_local(local_img_path: str, data: S3Object):
    local_img = _load_image(local_img_path)
    s3_img = Image.open(data)
    assert local_img == s3_img


def _verify_image_iterable_dataset(
    img_path_prefix: str,
    dataset: S3IterableDataset,
):
    assert dataset is not None
    assert img_path_prefix
    for index, data in enumerate(dataset):
        assert data is not None
        _compare_dataset_img_against_local(f"{img_path_prefix}img{index}.jpg", data)


def _pytorch_dataloader(
    dataset: Dataset, batch_size: int = 1
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=0, shuffle=False
    )


def _map_image_to_tensor(img: Image) -> Tensor:
    to_tensor = torchvision.transforms.ToTensor()
    return to_tensor(img)


def _create_local_dataloader(batch_size: int = 1, with_map_dataset: bool = False):
    full_path = os.path.join(ABSOLUTE_PATH, LOCAL_DATASET_RELATIVE_PATH)
    lister = torchdata.datapipes.iter.FileLister([full_path])
    dataset = torchdata.datapipes.iter.FileOpener(lister, mode="rb")
    if with_map_dataset:
        wrapper = IterableWrapper(enumerate(dataset))
        dataset = wrapper.to_map_datapipe()
    return _pytorch_dataloader(dataset.map(_load_image_as_tensor), batch_size)


def _load_image_as_tensor(sample: Tuple[str, StreamWrapper]) -> Tensor:
    sample_name, reader = sample
    return _map_image_to_tensor(_load_image(sample_name))


def _get_dataloader_len(dataloader: DataLoader):
    if isinstance(dataloader.dataset, MapDataPipe) or isinstance(
        dataloader.dataset, S3MapStyleDataset
    ):
        return len(dataloader)
    else:
        return len(list(dataloader))
