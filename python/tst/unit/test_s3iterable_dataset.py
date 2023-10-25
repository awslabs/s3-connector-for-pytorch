from typing import Iterable, Callable, Union, Sequence

import pytest

from s3dataset.s3iterable_dataset import S3IterableDataset
from s3dataset.s3object import S3Object
from test_s3dataset_base import (
    TEST_BUCKET,
    _create_mock_client_with_dummy_objects,
    S3_PREFIX,
    TEST_REGION,
)


def test_dataset_creation_from_objects_with_client_single_object():
    client = _create_mock_client_with_dummy_objects(TEST_BUCKET, ["single_object"])
    dataset = S3IterableDataset.from_objects(
        f"{S3_PREFIX}{TEST_BUCKET}/single_object", client=client
    )
    assert isinstance(dataset, S3IterableDataset)
    _test_s3iterable_dataset(
        dataset, ["single_object"], 1, lambda data: data.object_info is None
    )


@pytest.mark.parametrize(
    "keys, expected_keys, expected_count",
    [
        (["obj1"], ["obj1"], 1),
        (["obj1", "obj2", "obj3"], ["obj1", "obj2", "obj3"], 3),
        (["obj1", "obj2", "obj3", "test"], ["obj1", "obj2", "obj3", "test"], 4),
    ],
)
def test_s3iterable_dataset_creation_from_objects_with_client(
    keys: Iterable[str], expected_keys: Sequence[str], expected_count: int
):
    client = _create_mock_client_with_dummy_objects(TEST_BUCKET, keys)
    object_uris = [f"{S3_PREFIX}{TEST_BUCKET}/{key}" for key in keys]
    dataset = S3IterableDataset.from_objects(object_uris, client=client)
    assert isinstance(dataset, S3IterableDataset)
    _test_s3iterable_dataset(
        dataset, expected_keys, expected_count, lambda data: data.object_info is None
    )


@pytest.mark.parametrize(
    "keys, prefix, expected_keys, expected_count",
    [
        (["obj1"], "", ["obj1"], 1),
        (["obj1", "obj2", "obj3"], None, ["obj1", "obj2", "obj3"], 3),
        (["obj1", "obj2", "obj3", "test"], "obj", ["obj1", "obj2", "obj3"], 3),
    ],
)
def test_s3iterable_dataset_creation_from_bucket_with_client(
    keys: Iterable[str], prefix: str, expected_keys: Sequence[str], expected_count: int
):
    client = _create_mock_client_with_dummy_objects(TEST_BUCKET, keys)
    dataset = S3IterableDataset.from_bucket(TEST_BUCKET, prefix=prefix, client=client)
    assert isinstance(dataset, S3IterableDataset)
    _test_s3iterable_dataset(
        dataset,
        expected_keys,
        expected_count,
        lambda data: data.object_info is not None,
    )


def test_s3iterable_dataset_creation_from_bucket_with_region():
    dataset = S3IterableDataset.from_bucket(TEST_BUCKET, region=TEST_REGION)
    assert isinstance(dataset, S3IterableDataset)
    assert dataset.region == TEST_REGION


@pytest.mark.parametrize(
    "keys",
    [
        ([]),
        ("single_object"),
        (["obj1", "obj2", "test"]),
    ],
)
def test_s3iterable_dataset_creation_from_objects_with_region(
    keys: Union[str, Iterable[str]]
):
    object_uris = [f"{S3_PREFIX}{TEST_BUCKET}/{key}" for key in keys]
    dataset = S3IterableDataset.from_objects(object_uris, region=TEST_REGION)
    assert isinstance(dataset, S3IterableDataset)
    assert dataset.region == TEST_REGION


def _test_s3iterable_dataset(
    dataset: S3IterableDataset,
    expected_keys: Sequence[str],
    expected_count: int,
    object_info_check: Callable[[S3Object], bool],
):
    for index, data in enumerate(dataset):
        assert data is not None
        assert data.bucket == TEST_BUCKET
        assert data.key == expected_keys[index]
        assert object_info_check(data)
        assert data._stream is None
        data.prefetch()
        assert data._stream is not None
        for content in data._stream:
            expected_content = (
                f"{TEST_BUCKET}-{expected_keys[index]}-dummyData".encode()
            )
            assert content == expected_content
    assert index + 1 == expected_count
