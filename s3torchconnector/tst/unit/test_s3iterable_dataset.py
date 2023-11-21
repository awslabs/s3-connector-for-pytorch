#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from typing import Iterable, Callable, Sequence, Any

import pytest

from s3torchconnector import S3IterableDataset
from s3torchconnector._s3client import S3Reader

from test_s3dataset_common import (
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
    "datasetLambda",
    [
        lambda: S3IterableDataset.from_objects([], region=None),
        lambda: S3IterableDataset.from_objects([], region=123),
        lambda: S3IterableDataset.from_prefix("s3://bucket/prefix", region=None),
        lambda: S3IterableDataset.from_prefix("s3://bucket/prefix", region=123),
    ],
)
def test_dataset_creation_fails_without_region(
    datasetLambda: Callable[[], S3IterableDataset]
):
    with pytest.raises(TypeError) as e:
        datasetLambda()
    assert e.value.args == ("Region must be a string",)


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
    _verify_dataset(
        dataset, expected_keys, lambda data: data._get_object_info is not None
    )


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
        lambda data: data._object_info is not None,
    )


@pytest.mark.parametrize(
    "key, transform, expected",
    [
        (
            "obj1",
            lambda s3reader: s3reader.read(),
            f"{TEST_BUCKET}-obj1-dummyData".encode(),
        ),
        (
            "obj1",
            lambda s3reader: s3reader.read().upper(),
            f"{TEST_BUCKET}-obj1-dummyData".upper().encode(),
        ),
        (
            "obj1",
            lambda s3reader: 2,
            2,
        ),
    ],
)
def test_transform_from_prefix(
    key: str, transform: Callable[[S3Reader], Any], expected: Any
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
            lambda s3reader: s3reader.read(),
            f"{TEST_BUCKET}-obj1-dummyData".encode(),
        ),
        (
            "obj1",
            lambda s3reader: s3reader.read().upper(),
            f"{TEST_BUCKET}-obj1-dummyData".upper().encode(),
        ),
        (
            "obj1",
            lambda s3reader: 2,
            2,
        ),
    ],
)
def test_transform_from_objects(
    key: str, transform: Callable[[S3Reader], Any], expected: Any
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
    _verify_dataset(dataset, expected_keys, lambda data: data._object_info is not None)
    _verify_dataset(dataset, expected_keys, lambda data: data._object_info is not None)


def _verify_dataset(
    dataset: S3IterableDataset,
    expected_keys: Sequence[str],
    object_info_check: Callable[[S3Reader], bool],
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
