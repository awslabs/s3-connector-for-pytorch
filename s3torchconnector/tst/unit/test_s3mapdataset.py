#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
import logging
from io import SEEK_END
from typing import Sequence, Callable, Any, Iterable
from unittest.mock import patch, MagicMock

import pytest

from s3torchconnector import S3MapDataset, S3Reader
from s3torchconnector._s3client import MockS3Client

from .test_s3dataset_common import (
    TEST_BUCKET,
    TEST_REGION,
    _create_mock_client_with_dummy_objects,
    S3_PREFIX,
    TEST_ENDPOINT,
)


def test_dataset_creation_from_prefix_with_region(caplog):
    with caplog.at_level(logging.INFO):
        dataset = S3MapDataset.from_prefix(S3_PREFIX, region=TEST_REGION)
    assert isinstance(dataset, S3MapDataset)
    assert dataset.region == TEST_REGION
    assert "Building S3MapDataset from_prefix" in caplog.text


def test_dataset_creation_from_objects_with_region(caplog):
    with caplog.at_level(logging.INFO):
        dataset = S3MapDataset.from_objects([], region=TEST_REGION)
    assert isinstance(dataset, S3MapDataset)
    assert dataset.region == TEST_REGION
    assert "Building S3MapDataset from_objects" in caplog.text


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


def test_from_prefix_seek_no_head():
    dataset = S3MapDataset.from_prefix(S3_PREFIX, region=TEST_REGION)

    # use mock client for unit testing
    client = _create_mock_client_with_dummy_objects(TEST_BUCKET, ["foo"])
    dataset._client = client

    with patch.object(
        MockS3Client, "head_object", wraps=client.head_object
    ) as head_object:
        s3_object = next(iter(dataset))
        s3_object.seek(0, SEEK_END)
    head_object.assert_not_called()


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


@pytest.mark.parametrize(
    "keys, expected_keys, worker_id, num_workers, rank, world_size",
    [
        # only one node is used
        ([], [], 0, 4, 0, 1),
        ([], [], 2, 3, 0, 1),
        (["obj1"], ["obj1"], 0, 2, 0, 1),
        (["obj1"], [], 1, 2, 0, 1),
        (["obj1", "obj2", "obj3"], ["obj1", "obj3"], 0, 2, 0, 1),
        (
            ["obj1", "obj2", "obj3", "obj4", "obj5"],
            ["obj1", "obj3", "obj5"],
            0,
            2,
            0,
            1,
        ),
        (["obj1", "obj2", "obj3", "test"], ["obj2", "test"], 1, 2, 0, 1),
        (["obj1", "obj2", "obj3"], ["obj2"], 1, 3, 0, 1),
        (["obj1", "obj2", "obj3", "obj4", "obj5"], ["obj1", "obj4"], 0, 3, 0, 1),
        (["obj1", "obj2", "obj3", "obj4", "obj5"], ["obj2", "obj5"], 1, 3, 0, 1),
        (["obj1", "obj2", "obj3", "obj4", "obj5"], ["obj3"], 2, 3, 0, 1),
        # two nodes are in use
        ([], [], 0, 4, 0, 2),
        ([], [], 2, 3, 0, 2),
        (["obj1"], ["obj1"], 0, 2, 0, 2),
        (["obj1"], [], 1, 1, 1, 2),
        (["obj1", "obj2", "obj3"], ["obj3"], 0, 2, 1, 2),
        (["obj1", "obj2", "obj3", "obj4", "obj5"], ["obj1", "obj5"], 0, 2, 0, 2),
        (
            ["obj1", "obj2", "obj3", "obj4", "obj5", "obj6", "obj7", "test"],
            ["obj4", "test"],
            1,
            2,
            1,
            2,
        ),
        (["obj1", "obj2", "obj3"], ["obj2"], 1, 3, 0, 2),
        (
            ["obj1", "obj2", "obj3", "obj4", "obj5", "obj6", "obj7"],
            ["obj1", "obj7"],
            0,
            3,
            0,
            2,
        ),
        (
            [
                "obj1",
                "obj2",
                "obj3",
                "obj4",
                "obj5",
                "obj6",
                "obj7",
                "obj8",
                "obj9",
                "obj10",
                "obj11",
                "obj12",
            ],
            ["obj5", "obj11"],
            1,
            3,
            1,
            2,
        ),
        (["obj1", "obj2", "obj3", "obj4", "obj5"], ["obj3"], 2, 3, 0, 2),
    ],
)
@patch("torch.utils.data.get_worker_info")
def test_dataset_creation_from_objects_against_multiple_workers(
    get_worker_info_mock,
    keys: Iterable[str],
    expected_keys: Sequence[str],
    worker_id: int,
    num_workers: int,
    rank: int,
    world_size: int,
):
    worker_info_mock = MagicMock(id=worker_id, num_workers=num_workers)
    get_worker_info_mock.return_value = worker_info_mock

    object_uris = [f"{S3_PREFIX}/{key}" for key in keys]
    dataset = S3MapDataset.from_objects(
        object_uris, region=TEST_REGION, rank=rank, world_size=world_size
    )

    # use mock client for unit testing
    client = _create_mock_client_with_dummy_objects(TEST_BUCKET, keys)
    dataset._client = client

    assert isinstance(dataset, S3MapDataset)
    assert len(dataset) == len(keys)
    for index, key in enumerate(keys):
        verify_item(dataset, index, key)


@pytest.mark.parametrize(
    "keys, prefix, expected_keys, worker_id, num_workers, rank, world_size",
    [
        # only one node is used
        ([], S3_PREFIX, [], 0, 4, 0, 1),
        ([], S3_PREFIX, [], 2, 3, 0, 1),
        (["obj1"], S3_PREFIX, ["obj1"], 0, 2, 0, 1),
        (["obj1"], f"{S3_PREFIX}/", [], 1, 2, 0, 1),
        (["obj1", "obj2", "obj3"], S3_PREFIX, ["obj1", "obj3"], 0, 2, 0, 1),
        (
            ["obj1", "obj2", "obj3", "obj4", "obj5"],
            f"{S3_PREFIX}/",
            ["obj1", "obj3", "obj5"],
            0,
            2,
            0,
            1,
        ),
        (["obj1", "obj2", "obj3", "test"], S3_PREFIX, ["obj2", "test"], 1, 2, 0, 1),
        (["obj1", "obj2", "obj3"], S3_PREFIX, ["obj2"], 1, 3, 0, 1),
        (
            ["obj1", "obj2", "obj3", "obj4", "obj5"],
            f"{S3_PREFIX}/",
            ["obj1", "obj4"],
            0,
            3,
            0,
            1,
        ),
        (
            ["obj1", "obj2", "obj3", "obj4", "obj5"],
            S3_PREFIX,
            ["obj2", "obj5"],
            1,
            3,
            0,
            1,
        ),
        (["obj1", "obj2", "obj3", "obj4", "obj5"], S3_PREFIX, ["obj3"], 2, 3, 0, 1),
        (
            ["obj1", "test1", "obj2", "obj3", "test2", "obj4", "obj5", "test4"],
            f"{S3_PREFIX}/obj",
            ["obj1", "obj4"],
            0,
            3,
            0,
            1,
        ),
        (
            [
                "test0",
                "obj1",
                "obj2",
                "obj3",
                "test1",
                "test2",
                "test3",
                "obj4",
                "obj5",
                "test4",
                "test5",
            ],
            f"{S3_PREFIX}/obj",
            ["obj2", "obj5"],
            1,
            3,
            0,
            1,
        ),
        # two nodes are in use
        ([], S3_PREFIX, [], 0, 4, 0, 2),
        ([], S3_PREFIX, [], 2, 3, 1, 2),
        (["obj1"], S3_PREFIX, ["obj1"], 0, 2, 0, 2),
        (["obj1"], f"{S3_PREFIX}/", [], 1, 2, 0, 2),
        (
            ["obj1", "obj2", "obj3", "obj4", "obj5"],
            S3_PREFIX,
            ["obj1", "obj5"],
            0,
            2,
            0,
            2,
        ),
        (
            ["obj1", "obj2", "obj3", "obj4", "obj5", "obj6", "obj7", "obj8"],
            f"{S3_PREFIX}/",
            ["obj3", "obj7"],
            0,
            2,
            1,
            2,
        ),
        (
            ["obj1", "obj2", "obj3", "obj4", "obj5", "test"],
            S3_PREFIX,
            ["obj2", "test"],
            1,
            2,
            0,
            2,
        ),
        (["obj1", "obj2", "obj3"], S3_PREFIX, ["obj2"], 1, 3, 0, 2),
        (
            ["obj1", "obj2", "obj3", "obj4", "obj5"],
            f"{S3_PREFIX}/",
            ["obj1"],
            0,
            3,
            0,
            2,
        ),
        (["obj1", "obj2", "obj3", "obj4", "obj5"], S3_PREFIX, ["obj5"], 1, 3, 1, 2),
        (["obj1", "obj2", "obj3", "obj4", "obj5"], S3_PREFIX, ["obj3"], 0, 1, 2, 3),
        (
            ["obj1", "test1", "obj2", "obj3", "test2", "obj4", "obj5", "test4"],
            f"{S3_PREFIX}/obj",
            ["obj1", "obj5"],
            0,
            2,
            0,
            2,
        ),
        (
            [
                "test0",
                "obj1",
                "obj2",
                "obj3",
                "test1",
                "test2",
                "test3",
                "obj4",
                "obj5",
                "test4",
                "test5",
            ],
            f"{S3_PREFIX}/obj",
            ["obj2", "obj5"],
            0,
            1,
            1,
            3,
        ),
    ],
)
@patch("torch.utils.data.get_worker_info")
def test_dataset_creation_from_prefix_against_multiple_workers(
    get_worker_info_mock,
    keys: Iterable[str],
    prefix: str,
    expected_keys: Sequence[str],
    worker_id: int,
    num_workers: int,
    rank: int,
    world_size: int,
):
    worker_info_mock = MagicMock(id=worker_id, num_workers=num_workers)
    get_worker_info_mock.return_value = worker_info_mock

    dataset = S3MapDataset.from_prefix(
        s3_uri=prefix, region=TEST_REGION, rank=rank, world_size=world_size
    )
    client = _create_mock_client_with_dummy_objects(TEST_BUCKET, keys)
    dataset._client = client
    assert isinstance(dataset, S3MapDataset)
    assert len(dataset) == len(expected_keys)

    for index, key in enumerate(expected_keys):
        verify_item(dataset, index, key)


def verify_item(dataset: S3MapDataset, index: int, expected_key: str):
    data = dataset[index]

    assert data is not None
    assert data.bucket == TEST_BUCKET
    assert data.key == expected_key
    assert data._stream is None
    expected_content = f"{TEST_BUCKET}-{expected_key}-dummyData".encode()
    content = data.read()
    assert content == expected_content
