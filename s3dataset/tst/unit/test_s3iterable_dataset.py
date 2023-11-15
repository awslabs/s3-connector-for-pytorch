from typing import Iterable, Callable, Sequence, Any

import pytest

from s3dataset import S3IterableDataset
from s3dataset_s3_client import S3Object
from test_s3dataset_base import (
    TEST_BUCKET,
    _create_mock_client_with_dummy_objects,
    S3_PREFIX,
    TEST_REGION,
)


def test_dataset_creation_from_prefix_with_region():
    dataset = S3IterableDataset.from_prefix(S3_PREFIX, region=TEST_REGION)
    assert isinstance(dataset, S3IterableDataset)
    assert dataset.region == TEST_REGION


def test_dataset_creation_from_objects_with_region():
    dataset = S3IterableDataset.from_objects([], region=TEST_REGION)
    assert isinstance(dataset, S3IterableDataset)
    assert dataset.region == TEST_REGION


@pytest.mark.parametrize(
    "keys, expected_keys",
    [
        ([], []),
        (["obj1"], ["obj1"]),
        (["obj1", "obj2", "obj3"], ["obj1", "obj2", "obj3"]),
        (["obj1", "obj2", "obj3", "test"], ["obj1", "obj2", "obj3", "test"]),
    ],
)
def test_dataset_creation_from_objects(
    keys: Iterable[str],
    expected_keys: Sequence[str],
):
    object_uris = [f"{S3_PREFIX}/{key}" for key in keys]
    dataset = S3IterableDataset.from_objects(object_uris, region=TEST_REGION)

    # use mock client for unit testing
    client = _create_mock_client_with_dummy_objects(TEST_BUCKET, keys)
    dataset._client = client

    assert isinstance(dataset, S3IterableDataset)
    _verify_dataset(dataset, expected_keys, lambda data: data.object_info is None)


@pytest.mark.parametrize(
    "keys, prefix, expected_keys",
    [
        ([], S3_PREFIX, []),
        (["obj1"], S3_PREFIX, ["obj1"]),
        (["obj1", "obj2", "obj3"], S3_PREFIX, ["obj1", "obj2", "obj3"]),
        (["obj1", "obj2", "obj3"], f"{S3_PREFIX}/", ["obj1", "obj2", "obj3"]),
        (
            ["obj1", "obj2", "obj3", "test"],
            f"{S3_PREFIX}/obj",
            ["obj1", "obj2", "obj3"],
        ),
    ],
)
def test_dataset_creation_from_prefix(
    keys: Iterable[str], prefix: str, expected_keys: Sequence[str]
):
    dataset = S3IterableDataset.from_prefix(s3_uri=prefix, region=TEST_REGION)

    # use mock client for unit testing
    client = _create_mock_client_with_dummy_objects(TEST_BUCKET, keys)
    dataset._client = client

    assert isinstance(dataset, S3IterableDataset)
    _verify_dataset(
        dataset,
        expected_keys,
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
def test_transform_from_prefix(
    key: str, transform: Callable[[S3Object], Any], expected: Any
):
    dataset = S3IterableDataset.from_prefix(
        S3_PREFIX,
        region=TEST_REGION,
        transform=transform,
    )

    # use mock client for unit testing
    client = _create_mock_client_with_dummy_objects(TEST_BUCKET, [key])
    dataset._client = client

    assert isinstance(dataset, S3IterableDataset)
    assert list(dataset) == [expected]


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
def test_transform_from_objects(
    key: str, transform: Callable[[S3Object], Any], expected: Any
):
    object_uris = f"{S3_PREFIX}/{key}"

    dataset = S3IterableDataset.from_objects(
        object_uris,
        region=TEST_REGION,
        transform=transform,
    )

    # use mock client for unit testing
    client = _create_mock_client_with_dummy_objects(TEST_BUCKET, [key])
    dataset._client = client

    assert isinstance(dataset, S3IterableDataset)
    assert list(dataset) == [expected]


@pytest.mark.parametrize(
    "keys, prefix, expected_keys",
    [
        ([], S3_PREFIX, []),
        (["obj1"], S3_PREFIX, ["obj1"]),
        (["obj1", "obj2", "obj3"], f"{S3_PREFIX}/", ["obj1", "obj2", "obj3"]),
        (
            ["obj1", "obj2", "obj3", "test"],
            f"{S3_PREFIX}/obj",
            ["obj1", "obj2", "obj3"],
        ),
    ],
)
def test_iteration_multiple_times(
    keys: Iterable[str], prefix: str, expected_keys: Sequence[str]
):
    dataset = S3IterableDataset.from_prefix(prefix, region=TEST_REGION)

    # use mock client for unit testing
    client = _create_mock_client_with_dummy_objects(TEST_BUCKET, keys)
    dataset._client = client

    # Test that we can iterate over dataset multiple times by verifying it multiple times.
    _verify_dataset(dataset, expected_keys, lambda data: data.object_info is not None)
    _verify_dataset(dataset, expected_keys, lambda data: data.object_info is not None)


def _verify_dataset(
    dataset: S3IterableDataset,
    expected_keys: Sequence[str],
    object_info_check: Callable[[S3Object], bool],
    *,
    times_to_verify: int = 2,
):
    for _ in range(times_to_verify):
        count = 0
        for index, data in enumerate(dataset):
            count += 1
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
        assert count == len(expected_keys)
