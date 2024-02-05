#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from typing import Sequence, Callable, Any

import pytest

from s3torchconnector import S3MapDataset, S3Reader

from test_s3dataset_common import (
    TEST_BUCKET,
    TEST_REGION,
    _create_mock_client_with_dummy_objects,
    S3_PREFIX,
    TEST_ENDPOINT,
)


def test_dataset_creation_from_prefix_with_region():
    dataset = S3MapDataset.from_prefix(S3_PREFIX, region=TEST_REGION)
    assert isinstance(dataset, S3MapDataset)
    assert dataset.region == TEST_REGION


def test_dataset_creation_from_objects_with_region():
    dataset = S3MapDataset.from_objects([], region=TEST_REGION)
    assert isinstance(dataset, S3MapDataset)
    assert dataset.region == TEST_REGION


@pytest.mark.parametrize(
    "keys",
    [
        ["obj1"],
        ["obj1", "obj2", "obj3"],
        ["obj1", "obj2", "obj3", "test"],
    ],
)
def test_dataset_creation_from_objects(keys: Sequence[str]):
    object_uris = [f"{S3_PREFIX}/{key}" for key in keys]
    dataset = S3MapDataset.from_objects(object_uris, region=TEST_REGION)

    # use mock client for unit testing
    client = _create_mock_client_with_dummy_objects(TEST_BUCKET, keys)
    dataset._client = client

    assert isinstance(dataset, S3MapDataset)
    assert len(dataset) == len(keys)
    for index, key in enumerate(keys):
        verify_item(dataset, index, key)


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
    keys: Sequence[str],
    prefix: str,
    expected_keys: Sequence[str],
):
    dataset = S3MapDataset.from_prefix(s3_uri=prefix, region=TEST_REGION)
    client = _create_mock_client_with_dummy_objects(TEST_BUCKET, keys)
    dataset._client = client
    assert isinstance(dataset, S3MapDataset)
    assert len(dataset) == len(expected_keys)

    for index, key in enumerate(expected_keys):
        verify_item(dataset, index, key)


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
    dataset = S3MapDataset.from_prefix(
        s3_uri=S3_PREFIX,
        region=TEST_REGION,
        transform=transform,
    )

    # use mock client for unit testing
    client = _create_mock_client_with_dummy_objects(TEST_BUCKET, [key])
    dataset._client = client

    assert isinstance(dataset, S3MapDataset)
    assert dataset[0] == expected


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

    dataset = S3MapDataset.from_objects(
        object_uris,
        region=TEST_REGION,
        transform=transform,
    )

    # use mock client for unit testing
    client = _create_mock_client_with_dummy_objects(TEST_BUCKET, [key])
    dataset._client = client

    assert isinstance(dataset, S3MapDataset)
    assert list(dataset) == [expected]


@pytest.mark.parametrize(
    "keys, length",
    [
        (["obj1"], 1),
        (["obj1", "obj2", "obj3"], 3),
        (["obj1", "obj2", "obj3", "test"], 4),
    ],
)
def test_call_len_twice(keys: Sequence[str], length: int):
    object_uris = (f"{S3_PREFIX}{TEST_BUCKET}/{key}" for key in keys)
    dataset = S3MapDataset.from_objects(object_uris, region=TEST_REGION)

    client = _create_mock_client_with_dummy_objects(TEST_BUCKET, keys)
    dataset._client = client

    assert isinstance(dataset, S3MapDataset)
    assert len(dataset) == length
    assert len(dataset) == length


def test_dataset_creation_from_prefix_with_region_and_endpoint():
    dataset = S3MapDataset.from_prefix(
        S3_PREFIX, region=TEST_REGION, endpoint=TEST_ENDPOINT
    )
    assert isinstance(dataset, S3MapDataset)
    assert dataset.endpoint == TEST_ENDPOINT


def verify_item(dataset: S3MapDataset, index: int, expected_key: str):
    data = dataset[index]

    assert data is not None
    assert data.bucket == TEST_BUCKET
    assert data.key == expected_key
    assert data._stream is None
    expected_content = f"{TEST_BUCKET}-{expected_key}-dummyData".encode()
    content = data.read()
    assert content == expected_content
