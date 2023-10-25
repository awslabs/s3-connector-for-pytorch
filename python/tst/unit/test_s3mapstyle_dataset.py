import pytest
from typing import Union, Iterable, Sequence

from s3dataset.s3mapstyle_dataset import S3MapStyleDataset
from test_s3dataset_base import (
    TEST_BUCKET,
    TEST_REGION,
    _create_mock_client_with_dummy_objects,
    S3_PREFIX,
)


def test_s3mapstyle_dataset_creation_from_objects_with_client_single_object():
    client = _create_mock_client_with_dummy_objects(TEST_BUCKET, ["single_object"])
    dataset = S3MapStyleDataset.from_objects(
        f"{S3_PREFIX}{TEST_BUCKET}/single_object", client=client
    )
    assert isinstance(dataset, S3MapStyleDataset)
    assert len(dataset) == 1
    _test_s3mapstyle_dataset(dataset, 0, "single_object")


@pytest.mark.parametrize(
    "keys",
    [
        [],
        "single_object",
        ["obj1", "obj2", "test"],
    ],
)
def test_s3mapstyle_dataset_creation_from_objects_with_region(
    keys: Union[str, Iterable[str]]
):
    object_uris = [f"{S3_PREFIX}{TEST_BUCKET}/{key}" for key in keys]
    dataset = S3MapStyleDataset.from_objects(object_uris, region=TEST_REGION)
    assert isinstance(dataset, S3MapStyleDataset)
    assert dataset.region == TEST_REGION
    assert len(dataset) == len(keys)


@pytest.mark.parametrize(
    "keys",
    [
        ["obj1"],
        ["obj1", "obj2", "obj3"],
        ["obj1", "obj2", "obj3", "test"],
    ],
)
def test_s3mapstyle_dataset_creation_from_objects_with_client(keys: Sequence[str]):
    client = _create_mock_client_with_dummy_objects(TEST_BUCKET, keys)
    object_uris = [f"{S3_PREFIX}{TEST_BUCKET}/{key}" for key in keys]
    dataset = S3MapStyleDataset.from_objects(object_uris, client=client)
    assert isinstance(dataset, S3MapStyleDataset)
    assert len(dataset) == len(keys)
    for index, key in enumerate(keys):
        _test_s3mapstyle_dataset(dataset, index, key)


def test_s3mapstyle_dataset_creation_from_bucket_with_region():
    dataset = S3MapStyleDataset.from_bucket(TEST_BUCKET, region=TEST_REGION)
    assert isinstance(dataset, S3MapStyleDataset)
    assert dataset.region == TEST_REGION


@pytest.mark.parametrize(
    "keys, prefix",
    [
        (["obj1"], ""),
        (["obj1", "obj2", "obj3"], None),
        (["obj1", "obj2", "obj3", "test"], "obj"),
        (["another", "test2", "obj1", "obj2", "obj3", "test"], "obj"),
    ],
)
def test_s3mapstyle_dataset_creation_from_bucket_with_client(
    keys: Sequence[str],
    prefix: str,
):
    client = _create_mock_client_with_dummy_objects(TEST_BUCKET, keys)
    dataset = S3MapStyleDataset.from_bucket(TEST_BUCKET, prefix=prefix, client=client)
    expected_keys = [key for key in keys if key.startswith(prefix or "")]
    assert isinstance(dataset, S3MapStyleDataset)
    assert len(dataset) == len(expected_keys)
    for index, key in enumerate(expected_keys):
        _test_s3mapstyle_dataset(dataset, index, key)


def _test_s3mapstyle_dataset(dataset: S3MapStyleDataset, index: int, expected_key: str):
    data = dataset[index]

    assert data is not None
    assert data.bucket == TEST_BUCKET
    assert data.key == expected_key
    assert data._stream is None
    expected_content = f"{TEST_BUCKET}-{expected_key}-dummyData".encode()
    content = data.read()
    assert content == expected_content
