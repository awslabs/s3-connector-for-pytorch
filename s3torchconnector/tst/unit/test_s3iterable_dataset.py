#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
import logging
from io import SEEK_END
from typing import Iterable, Callable, Sequence, Any

import pytest
import hypothesis.strategies as st
from hypothesis import given, assume
from unittest.mock import patch, MagicMock

from s3torchconnector import S3IterableDataset, S3Reader, S3ReaderConstructor
from s3torchconnector.s3reader import S3ReaderConstructorProtocol
from s3torchconnector._s3client import MockS3Client

from .test_s3dataset_common import (
    TEST_BUCKET,
    _create_mock_client_with_dummy_objects,
    S3_PREFIX,
    TEST_REGION,
    TEST_ENDPOINT,
    READER_TYPE_STRING_TO_CLASS,
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


def test_default_reader_type_from_prefix():
    dataset = S3IterableDataset.from_prefix(S3_PREFIX, region=TEST_REGION)
    reader_type_string = S3ReaderConstructor.get_reader_type_string(
        dataset._reader_constructor
    )
    assert reader_type_string == "sequential"


def test_default_reader_type_from_objects():
    dataset = S3IterableDataset.from_objects([], region=TEST_REGION)
    reader_type_string = S3ReaderConstructor.get_reader_type_string(
        dataset._reader_constructor
    )
    assert reader_type_string == "sequential"


def test_dataset_creation_from_prefix_with_reader_constructor(
    reader_constructor: S3ReaderConstructorProtocol,
):
    dataset = S3IterableDataset.from_prefix(
        S3_PREFIX,
        region=TEST_REGION,
        reader_constructor=reader_constructor,
    )
    assert dataset._reader_constructor == reader_constructor


def test_dataset_creation_from_objects_with_reader_constructor(
    reader_constructor: S3ReaderConstructorProtocol,
):
    dataset = S3IterableDataset.from_objects(
        [],
        region=TEST_REGION,
        reader_constructor=reader_constructor,
    )
    assert dataset._reader_constructor == reader_constructor


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
    reader_constructor: S3ReaderConstructorProtocol,
):
    object_uris = [f"{S3_PREFIX}/{key}" for key in keys]
    dataset = S3IterableDataset.from_objects(
        object_uris, region=TEST_REGION, reader_constructor=reader_constructor
    )

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
    keys: Iterable[str],
    prefix: str,
    expected_keys: Sequence[str],
    reader_constructor: S3ReaderConstructorProtocol,
):
    dataset = S3IterableDataset.from_prefix(
        s3_uri=prefix, region=TEST_REGION, reader_constructor=reader_constructor
    )

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
    key: str,
    transform: Callable[[S3Reader], Any],
    expected: Any,
    reader_constructor: S3ReaderConstructorProtocol,
):
    dataset = S3IterableDataset.from_prefix(
        S3_PREFIX,
        region=TEST_REGION,
        transform=transform,
        reader_constructor=reader_constructor,
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
    key: str,
    transform: Callable[[S3Reader], Any],
    expected: Any,
    reader_constructor: S3ReaderConstructorProtocol,
):
    object_uris = f"{S3_PREFIX}/{key}"

    dataset = S3IterableDataset.from_objects(
        object_uris,
        region=TEST_REGION,
        transform=transform,
        reader_constructor=reader_constructor,
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
    keys: Iterable[str],
    prefix: str,
    expected_keys: Sequence[str],
    reader_constructor: S3ReaderConstructorProtocol,
):
    dataset = S3IterableDataset.from_prefix(
        prefix, region=TEST_REGION, reader_constructor=reader_constructor
    )

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


def test_from_prefix_seek_no_head(reader_constructor: S3ReaderConstructorProtocol):
    dataset = S3IterableDataset.from_prefix(
        S3_PREFIX,
        region=TEST_REGION,
        reader_constructor=reader_constructor,
    )

    # use mock client for unit testing
    client = _create_mock_client_with_dummy_objects(TEST_BUCKET, ["foo"])
    dataset._client = client

    with patch.object(
        MockS3Client, "head_object", wraps=client.head_object
    ) as head_object:
        s3_object = next(iter(dataset))
        s3_object.seek(0, SEEK_END)
    head_object.assert_not_called()


# Strategies for generating test cases
keys_strategy = st.lists(st.text(), min_size=1, max_size=20, unique=True)
create_from_prefix_strategy = st.booleans()


# Composite strategy for num_workers_per_rank, based on world_size
@st.composite
def _num_workers_per_rank_strategy(draw):
    world_size = draw(st.integers(min_value=1, max_value=4))
    num_workers_per_rank = draw(
        st.lists(
            st.integers(min_value=1, max_value=4),
            min_size=world_size,
            max_size=world_size,
        )
    )
    return world_size, num_workers_per_rank


# Put patch decorators before @given and @parametrize
@patch("torch.distributed.get_world_size")
@patch("torch.distributed.get_rank")
@patch("torch.distributed.is_initialized")
@patch("torch.utils.data.get_worker_info")
@given(
    keys_for_prefix1=keys_strategy,
    keys_for_prefix2=keys_strategy,
    create_from_prefix=create_from_prefix_strategy,
    world_size_and_num_workers=_num_workers_per_rank_strategy(),
)
def test_dataset_creation_against_multiple_workers(
    get_worker_info_mock,
    is_initialized_mock,
    get_rank_mock,
    get_world_size_mock,
    keys_for_prefix1,
    keys_for_prefix2,
    create_from_prefix,
    world_size_and_num_workers,
    reader_constructor: S3ReaderConstructorProtocol,
):
    """Test the iterating over S3IterableDataset with different numbers of ranks/workers when sharding is enabled.

    Args:
        get_worker_info_mock (MagicMock): Mock for torch.utils.data.get_worker_info().
        is_initialized_mock (MagicMock): Mock for torch.distributed.is_initialized().
        get_rank_mock (MagicMock): Mock for torch.distributed.get_rank().
        get_world_size_mock (MagicMock): Mock for torch.distributed.get_world_size().
        keys_for_prefix1 (list): A list of strings representing keys for the first prefix.
        keys_for_prefix2 (list): A list of strings representing keys for the second prefix, should be ignored with .from_prefix
        create_from_prefix (bool): Whether to create the dataset from a prefix or a list of object URIs.
        world_size_and_num_workers (tuple): A tuple containing world_size and num_workers for each rank.
        reader_type (ReaderType): S3Reader ReaderType
    """
    prefix1 = "obj"
    prefix2 = "test"
    world_size, num_workers_per_rank = world_size_and_num_workers

    # Assume valid input combinations
    assume(keys_for_prefix1 and keys_for_prefix2)
    assume(world_size >= 1 and world_size == len(num_workers_per_rank))

    all_keys_for_prefix1 = [f"{prefix1}/{key}" for key in keys_for_prefix1]
    all_keys = all_keys_for_prefix1 + [f"{prefix2}/{key}" for key in keys_for_prefix2]
    object_uris = [f"{S3_PREFIX}/{key}" for key in all_keys]

    is_initialized_mock.return_value = True
    get_world_size_mock.return_value = world_size

    all_keys_from_workers = []
    num_keys_per_rank = []
    for rank in range(world_size):
        get_rank_mock.return_value = rank
        num_workers = num_workers_per_rank[rank]
        """Gather all keys from all workers for a specific rank and them to all_keys_from_workers
        As ranks and world size initiated in the constructor of S3IterableDataset, we need to reset them
        for each rank
        """
        is_initialized_mock.return_value = True
        num_keys = 0
        for worker_id in range(num_workers):
            worker_info_mock = MagicMock(id=worker_id, num_workers=num_workers)
            get_worker_info_mock.return_value = worker_info_mock

            if create_from_prefix:
                dataset = S3IterableDataset.from_prefix(
                    s3_uri=f"{S3_PREFIX}/{prefix1}",
                    region=TEST_REGION,
                    enable_sharding=True,
                    reader_constructor=reader_constructor,
                )
            else:
                dataset = S3IterableDataset.from_objects(
                    object_uris=object_uris,
                    region=TEST_REGION,
                    enable_sharding=True,
                    reader_constructor=reader_constructor,
                )

            assert dataset._reader_constructor == reader_constructor

            client = _create_mock_client_with_dummy_objects(TEST_BUCKET, all_keys)
            dataset._client = client

            worker_keys = []
            for data in dataset:
                worker_keys.append(data.key)

            all_keys_from_workers.append(worker_keys)
            num_keys += len(worker_keys)
        num_keys_per_rank.append(num_keys)

    # Get list of all seen keys
    flattened_keys_from_workers = [
        key for nested_keys in all_keys_from_workers for key in nested_keys
    ]
    if create_from_prefix:
        assert len(set(flattened_keys_from_workers)) == len(
            all_keys_for_prefix1
        ), "All keys under prefix1 should be processed"
        assert set(flattened_keys_from_workers) == set(
            all_keys_for_prefix1
        ), "Union of keys under prefix1 should be the same"
        assert all(
            [key.startswith(prefix1) for key in flattened_keys_from_workers]
        ), "All keys should start with prefix1"
        expected_num_per_rank = len(keys_for_prefix1) // world_size
    else:
        assert len(set(flattened_keys_from_workers)) == len(
            all_keys
        ), "All keys should be processed"
        assert set(flattened_keys_from_workers) == set(
            all_keys
        ), "Union of keys should be the same"
        expected_num_per_rank = len(all_keys) // world_size
    assert all(
        [(num_keys - expected_num_per_rank) <= 1 for num_keys in num_keys_per_rank]
    ), "The number of keys should be evenly distributed across ranks, with a difference of at most 1"


def test_user_agent_includes_dataset_and_reader_type(
    reader_constructor: S3ReaderConstructorProtocol,
):
    """Test that user agent includes dataset type and reader type."""
    dataset = S3IterableDataset.from_prefix(
        S3_PREFIX, region=TEST_REGION, reader_constructor=reader_constructor
    )
    dataset._get_client()

    user_agent = dataset._client.user_agent_prefix

    reader_type_string = S3ReaderConstructor.get_reader_type_string(reader_constructor)
    # expect: sequential / range_based

    assert "md/dataset#iterable" in user_agent
    assert f"md/reader_type#{reader_type_string}" in user_agent


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
            reader_type_string = S3ReaderConstructor.get_reader_type_string(
                dataset._reader_constructor
            )
            assert isinstance(data, READER_TYPE_STRING_TO_CLASS[reader_type_string])
            assert object_info_check(data)
            expected_content = (
                f"{TEST_BUCKET}-{expected_keys[index]}-dummyData".encode()
            )
            content = data.read()
            assert content == expected_content
        assert count == len(expected_keys)
