#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
import logging
from io import SEEK_END
from typing import Iterable, Callable, Sequence, Any
from unittest.mock import patch, MagicMock

import pytest

from s3torchconnector import S3IterableDataset, S3Reader
from s3torchconnector._s3client import MockS3Client

from .test_s3dataset_common import (
    TEST_BUCKET,
    _create_mock_client_with_dummy_objects,
    S3_PREFIX,
    TEST_REGION,
    TEST_ENDPOINT,
)


def test_dataset_creation_from_prefix_with_region(caplog):
    with caplog.at_level(logging.INFO):
        dataset = S3IterableDataset.from_prefix(S3_PREFIX, region=TEST_REGION)
    assert isinstance(dataset, S3IterableDataset)
    assert dataset.region == TEST_REGION
    assert "Building S3IterableDataset from_prefix" in caplog.text


def test_dataset_creation_from_objects_with_region(caplog):
    with caplog.at_level(logging.INFO):
        dataset = S3IterableDataset.from_objects([], region=TEST_REGION)
    assert isinstance(dataset, S3IterableDataset)
    assert dataset.region == TEST_REGION
    assert "Building S3IterableDataset from_objects" in caplog.text


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


def test_dataset_creation_from_prefix_with_region_and_endpoint():
    dataset = S3IterableDataset.from_prefix(
        S3_PREFIX, region=TEST_REGION, endpoint=TEST_ENDPOINT
    )
    assert isinstance(dataset, S3IterableDataset)
    assert dataset.endpoint == TEST_ENDPOINT


def test_from_prefix_seek_no_head():
    dataset = S3IterableDataset.from_prefix(S3_PREFIX, region=TEST_REGION)

    # use mock client for unit testing
    client = _create_mock_client_with_dummy_objects(TEST_BUCKET, ["foo"])
    dataset._client = client

    with patch.object(
        MockS3Client, "head_object", wraps=client.head_object
    ) as head_object:
        s3_object = next(iter(dataset))
        s3_object.seek(0, SEEK_END)
    head_object.assert_not_called()


"""
    This test validates the distribution of object keys across workers and processes
    in a distributed data loading scenario, ensuring that each worker processes get
    the expected subset of keys.
    all_keys                  = List of object keys to be processed
    expected_keys             = Expected list of keys to be processed by the active worker
    worker_id                 = ID of the current worker thread/process within the process
    num_workers               = Total number of worker threads/processes within the process
    rank                      = Rank (index) of the current process within the world (group of processes)
    world_size                = Total number of processes in the world

    Legend:
    r{rank}w{worker}          = Worker {worker} in Rank {rank}
    [obj1, obj2]              = Objects assigned to the worker
    [obj1, obj2]<-active      = Active worker (objects being tested)
"""
@pytest.mark.parametrize(
    "all_keys, expected_keys, worker_id, num_workers, rank, world_size",
    [
        # only one node is used
        ([], [], 0, 4, 0, 1),
        # r0w0[]<-active  r0w1[]  r0w2[]  r0w3[]
        ([], [], 2, 3, 0, 1),
        # r0w0[]  r0w1[]  r0w2[]<-active
        (["obj1"], ["obj1"], 0, 2, 0, 1),
        # r0w0[obj1]<-active  r0w1[]
        (["obj1"], [], 1, 2, 0, 1),
        # r0w0[obj1]  r0w1[]<-active
        (["obj1", "obj2", "obj3"], ["obj1", "obj3"], 0, 2, 0, 1),
        # r0w0[obj1, obj3]<-active  r0w1[obj2]
        (
            ["obj1", "obj2", "obj3", "obj4", "obj5"],
            ["obj1", "obj3", "obj5"],
            0,
            2,
            0,
            1,
        ),
        # r0w0[obj1, obj3, obj5]<-active  r0w1[obj2, obj4]
        (["obj1", "obj2", "obj3", "test"], ["obj2", "test"], 1, 2, 0, 1),
        # r0w0[obj1, obj3, obj5]  r0w1[obj2, test]<-active
        (["obj1", "obj2", "obj3"], ["obj2"], 1, 3, 0, 1),
        # r0w0[obj1]  r0w1[obj2]<-active  r0w2[obj3]
        (["obj1", "obj2", "obj3", "obj4", "obj5"], ["obj1", "obj4"], 0, 3, 0, 1),
        # r0w0[obj1, obj4]<-active  r0w1[obj2, obj5]  r0w2[obj3]
        (["obj1", "obj2", "obj3", "obj4", "obj5"], ["obj2", "obj5"], 1, 3, 0, 1),
        # r0w0[obj1, obj4]  r0w1[obj2, obj5]<-active  r0w2[obj3]
        (["obj1", "obj2", "obj3", "obj4", "obj5"], ["obj3"], 2, 3, 0, 1),
        # r0w0[obj1, obj4]  r0w1[obj2, obj5]  r0w2[obj3]<-active
        # two nodes are in use
        ([], [], 0, 4, 0, 2),
        # r0w0[]<-active  r0w1[]  r0w2[]  r0w3[]        r1w0[]  r1w1[]  r1w2[]  r1w3[]
        ([], [], 2, 3, 0, 2),
        # r0w0[]  r0w1[]  r0w2[]<-active  r0w3[]        r1w0[]  r1w1[]  r1w2[]  r1w3[]
        (["obj1"], ["obj1"], 0, 2, 0, 2),
        # r0w0[obj1]<-active  r0w1[]                    r1w0[]  r1w1[]
        (["obj1"], [], 0, 1, 1, 2),
        # r0w0[obj1]  r1w0[]<-active
        (["obj1", "obj2", "obj3"], ["obj3"], 0, 2, 1, 2),
        # r0w0[obj1]  r0w1[obj2]                        r1w0[obj3]<-active  r1w1[]
        (["obj1", "obj2", "obj3", "obj4", "obj5"], ["obj1", "obj5"], 0, 2, 0, 2),
        # r0w0[obj1, obj5]<-active  r0w1[obj2]          r1w0[obj3]  r1w1[obj4]
        (
            ["obj1", "obj2", "obj3", "obj4", "obj5", "obj6", "obj7", "test"],
            ["obj4", "test"],
            1,
            2,
            1,
            2,
        ),
        # r0w0[obj1, obj5]  r0w1[obj2, obj6]            r1w0[obj3, obj7]  r1w1[obj4, test]<-active
        (["obj1", "obj2", "obj3"], ["obj2"], 1, 3, 0, 2),
        # r0w0[obj1]  r0w1[obj2]<-active r0w2[obj3]     r1w0[]  r1w1[] r1w2[]
        (
            ["obj1", "obj2", "obj3", "obj4", "obj5", "obj6", "obj7"],
            ["obj1", "obj7"],
            0,
            3,
            0,
            2,
        ),
        # r0w0[obj1, obj7]<-active  r0w1[obj2] r0w2[obj3]       r1w0[obj4]  r1w1[obj5] r1w2[obj6]
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
        # r0w0[obj1, obj7]  r0w1[obj2, obj8] r0w2[obj3, obj9]
        # r1w0[obj4, obj10]  r1w1[obj5, obj11]<-active r1w2[obj6, obj12]
        (["obj1", "obj2", "obj3", "obj4", "obj5"], ["obj3"], 2, 3, 0, 2),
        # r0w0[obj1]  r0w1[obj2] r0w2[obj3]<-active     r1w0[obj4]  r1w1[obj5] r1w2[]
    ],
)
@patch("torch.distributed.get_world_size")
@patch("torch.distributed.get_rank")
@patch("torch.distributed.is_initialized")
@patch("torch.utils.data.get_worker_info")
def test_dataset_creation_from_objects_against_multiple_workers(
    get_worker_info_mock,
    is_initialized_mock,
    get_rank_mock,
    get_world_size_mock,
    all_keys: Iterable[str],
    expected_keys: Sequence[str],
    worker_id: int,
    num_workers: int,
    rank: int,
    world_size: int,
):
    worker_info_mock = MagicMock(id=worker_id, num_workers=num_workers)
    get_worker_info_mock.return_value = worker_info_mock
    # assume torch.distributed is always initialized
    is_initialized_mock.return_value = True
    get_rank_mock.return_value = rank
    get_world_size_mock.return_value = world_size

    object_uris = [f"{S3_PREFIX}/{key}" for key in all_keys]
    dataset = S3IterableDataset.from_objects(object_uris, region=TEST_REGION)

    # use mock client for unit testing
    client = _create_mock_client_with_dummy_objects(TEST_BUCKET, all_keys)
    dataset._client = client

    assert isinstance(dataset, S3IterableDataset)
    _verify_dataset(
        dataset, expected_keys, lambda data: data._get_object_info is not None
    )


"""
    This test validates the distribution of object keys across workers and processes
    in a distributed data loading scenario, ensuring that each worker processes get
    the expected subset of keys.
    all_keys                  = List of object keys to be processed
    prefix                     = Prefix for the object keys
    expected_keys             = Expected list of keys to be processed by the active worker
    worker_id                 = ID of the current worker thread/process within the process
    num_workers               = Total number of worker threads/processes within the process
    rank                      = Rank (index) of the current process within the world (group of processes)
    world_size                = Total number of processes in the world

    Legend:
    r{rank}w{worker}          = Worker {worker} in Rank {rank}
    [obj1, obj2]              = Objects assigned to the worker
    [obj1, obj2]<-active      = Active worker (objects being tested)
"""
@pytest.mark.parametrize(
    "all_keys, prefix, expected_keys, worker_id, num_workers, rank, world_size",
    [
        # only one node is used
        ([], S3_PREFIX, [], 0, 4, 0, 1),
        # r0w0[]<-active  r0w1[]  r0w2[]  r0w3[]
        ([], S3_PREFIX, [], 2, 3, 0, 1),
        # r0w0[]  r0w1[]  r0w2[]<-active
        (["obj1"], S3_PREFIX, ["obj1"], 0, 2, 0, 1),
        # r0w0[obj1]<-active  r0w1[]
        (["obj1"], f"{S3_PREFIX}/", [], 1, 2, 0, 1),
        # r0w0[obj1]  r0w1[]<-active
        (["obj1", "obj2", "obj3"], S3_PREFIX, ["obj1", "obj3"], 0, 2, 0, 1),
        # r0w0[obj1, obj3]<-active  r0w1[obj2]
        (
            ["obj1", "obj2", "obj3", "obj4", "obj5"],
            f"{S3_PREFIX}/",
            ["obj1", "obj3", "obj5"],
            0,
            2,
            0,
            1,
        ),
        # r0w0[obj1, obj3, obj5]<-active  r0w1[obj2, obj4]
        (["obj1", "obj2", "obj3", "test"], S3_PREFIX, ["obj2", "test"], 1, 2, 0, 1),
        # r0w0[obj1, obj3, obj5]  r0w1[obj2, test]<-active
        (["obj1", "obj2", "obj3"], S3_PREFIX, ["obj2"], 1, 3, 0, 1),
        # r0w0[obj1]  r0w1[obj2]<-active  r0w2[obj3]
        (
            ["obj1", "obj2", "obj3", "obj4", "obj5"],
            f"{S3_PREFIX}/",
            ["obj1", "obj4"],
            0,
            3,
            0,
            1,
        ),
        # r0w0[obj1, obj4]<-active  r0w1[obj2, obj5]  r0w2[obj3]
        (
            ["obj1", "obj2", "obj3", "obj4", "obj5"],
            S3_PREFIX,
            ["obj2", "obj5"],
            1,
            3,
            0,
            1,
        ),
        # r0w0[obj1, obj4]  r0w1[obj2, obj5]<-active  r0w2[obj3]
        (["obj1", "obj2", "obj3", "obj4", "obj5"], S3_PREFIX, ["obj3"], 2, 3, 0, 1),
        # r0w0[obj1, obj4]  r0w1[obj2, obj5]  r0w2[obj3]<-active
        (
            ["obj1", "test1", "obj2", "obj3", "test2", "obj4", "obj5", "test4"],
            f"{S3_PREFIX}/obj",
            ["obj1", "obj4"],
            0,
            3,
            0,
            1,
        ),
        # r0w0[obj1, obj4]<-active  r0w1[obj2, obj5]  r0w2[obj3]
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
        # r0w0[obj1, obj4]  r0w1[obj2, obj5]<-active  r0w2[obj3]

        # two nodes are in use
        ([], S3_PREFIX, [], 0, 4, 0, 2),
        # r0w0[]<-active  r0w1[]  r0w2[]  r0w3[]        r1w0[]  r1w1[]  r1w2[]  r1w3[]
        ([], S3_PREFIX, [], 2, 3, 1, 2),
        # r0w0[]  r0w1[]  r0w2[]                        r1w0[]  r1w1[]  r1w2[]<-active  r1w3[]
        (["obj1"], S3_PREFIX, ["obj1"], 0, 2, 0, 2),
        # r0w0[obj1]<-active  r0w1[]                    r1w0[]  r1w1[]
        (["obj1"], f"{S3_PREFIX}/", [], 1, 2, 0, 2),
        # r0w0[obj1]  r0w1[]<-active                    r1w0[]  r1w1[]
        (
            ["obj1", "obj2", "obj3", "obj4", "obj5"],
            S3_PREFIX,
            ["obj1", "obj5"],
            0,
            2,
            0,
            2,
        ),
        # r0w0[obj1, obj5]<-active  r0w1[obj2]          r1w0[obj3]  r1w1[obj4]
        (
            ["obj1", "obj2", "obj3", "obj4", "obj5", "obj6", "obj7", "obj8"],
            f"{S3_PREFIX}/",
            ["obj3", "obj7"],
            0,
            2,
            1,
            2,
        ),
        # r0w0[obj1, obj5]  r0w1[obj2, obj6]            r1w0[obj3, obj7]<-active  r1w1[obj4, obj8]
        (
            ["obj1", "obj2", "obj3", "obj4", "obj5", "test"],
            S3_PREFIX,
            ["obj2", "test"],
            1,
            2,
            0,
            2,
        ),
        # r0w0[obj1, obj5]  r0w1[obj2, test]<-active                r1w0[obj3]  r1w1[obj4]
        (["obj1", "obj2", "obj3"], S3_PREFIX, ["obj2"], 1, 3, 0, 2),
        # r0w0[obj1]  r0w1[obj2]<-active  r0w2[obj3]                r1w0[]  r1w1[]  r1w2[]
        (
            ["obj1", "obj2", "obj3", "obj4", "obj5"],
            f"{S3_PREFIX}/",
            ["obj1"],
            0,
            3,
            0,
            2,
        ),
        # r0w0[obj1]<-active  r0w1[obj2]  r0w2[obj3]                r1w0[obj4]  r1w1[obj5]  r1w2[]
        (["obj1", "obj2", "obj3", "obj4", "obj5"], S3_PREFIX, ["obj5"], 1, 3, 1, 2),
        # r0w0[obj1]  r0w1[obj2]  r0w2[obj3]                        r1w0[obj4]  r1w1[obj5]<-active  r1w2[]
        (["obj1", "obj2", "obj3", "obj4", "obj5"], S3_PREFIX, ["obj3"], 0, 1, 2, 3),
        # r0w0[obj1, obj4]          r1w0[obj2, obj5]                r2w0[obj3]<-active
        (
            ["obj1", "test1", "obj2", "obj3", "test2", "obj4", "obj5", "test4"],
            f"{S3_PREFIX}/obj",
            ["obj1", "obj5"],
            0,
            2,
            0,
            2,
        ),
        # r0w0[obj1, obj5]<-active  r0w1[obj2]                      r1w0[obj3]  r1w1[obj4]
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
        # r0w0[obj1, obj4]          r1w0[obj2, obj5]<-active        r2w0[obj3]
    ],
)
@patch("torch.distributed.get_world_size")
@patch("torch.distributed.get_rank")
@patch("torch.distributed.is_initialized")
@patch("torch.utils.data.get_worker_info")
def test_dataset_creation_from_prefix_against_multiple_workers(
    get_worker_info_mock,
    is_initialized_mock,
    get_rank_mock,
    get_world_size_mock,
    all_keys: Iterable[str],
    prefix: str,
    expected_keys: Sequence[str],
    worker_id: int,
    num_workers: int,
    rank: int,
    world_size: int,
):
    worker_info_mock = MagicMock(id=worker_id, num_workers=num_workers)
    get_worker_info_mock.return_value = worker_info_mock
    # assume torch.distributed is initialized, only when world size is bigger then 1
    is_initialized_mock.return_value = world_size != 1
    get_rank_mock.return_value = rank
    get_world_size_mock.return_value = world_size

    dataset = S3IterableDataset.from_prefix(
        s3_uri=prefix,
        region=TEST_REGION,
    )

    # use mock client for unit testing
    client = _create_mock_client_with_dummy_objects(TEST_BUCKET, all_keys)
    dataset._client = client

    assert isinstance(dataset, S3IterableDataset)
    _verify_dataset(
        dataset,
        expected_keys,
        lambda data: data._object_info is not None,
    )


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
