from typing import Iterable, Callable, Union, Sequence, Any

import pytest

from s3dataset import S3IterableDataset
from s3dataset_s3_client import S3Object
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
    _verify_dataset(
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
def test_dataset_creation_from_objects_with_client(
    keys: Iterable[str], expected_keys: Sequence[str], expected_count: int
):
    client = _create_mock_client_with_dummy_objects(TEST_BUCKET, keys)
    object_uris = [f"{S3_PREFIX}{TEST_BUCKET}/{key}" for key in keys]
    dataset = S3IterableDataset.from_objects(object_uris, client=client)
    assert isinstance(dataset, S3IterableDataset)
    _verify_dataset(
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
def test_dataset_creation_from_bucket_with_client(
    keys: Iterable[str], prefix: str, expected_keys: Sequence[str], expected_count: int
):
    client = _create_mock_client_with_dummy_objects(TEST_BUCKET, keys)
    dataset = S3IterableDataset.from_bucket(TEST_BUCKET, prefix=prefix, client=client)
    assert isinstance(dataset, S3IterableDataset)
    _verify_dataset(
        dataset,
        expected_keys,
        expected_count,
        lambda data: data.object_info is not None,
    )


@pytest.mark.parametrize(
    "key, transform, expected",
    [
        (
            "obj1",
            lambda s3object: s3object.read(),
            f"{TEST_BUCKET}-obj1-dummyData".encode(),
        ),
        (
            "obj1",
            lambda s3object: s3object.read().upper(),
            f"{TEST_BUCKET}-obj1-dummyData".upper().encode(),
        ),
        (
            "obj1",
            lambda s3object: 2,
            2,
        ),
    ],
)
def test_dataset_creation_from_bucket_with_client(
    key: str, transform: Callable[[S3Object], Any], expected: Any
):
    client = _create_mock_client_with_dummy_objects(TEST_BUCKET, [key])
    dataset = S3IterableDataset.from_bucket(
        TEST_BUCKET,
        client=client,
        transform=transform,
    )
    assert isinstance(dataset, S3IterableDataset)
    assert list(dataset) == [expected]


def test_dataset_creation_from_bucket_with_region():
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
def test_dataset_creation_from_objects_with_region(keys: Union[str, Iterable[str]]):
    object_uris = [f"{S3_PREFIX}{TEST_BUCKET}/{key}" for key in keys]
    dataset = S3IterableDataset.from_objects(object_uris, region=TEST_REGION)
    assert isinstance(dataset, S3IterableDataset)
    assert dataset.region == TEST_REGION


def _verify_dataset(
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
