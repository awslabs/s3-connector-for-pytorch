import pickle
from datetime import timedelta
from typing import Iterable, Callable, Union, Sequence, Any, List

import hypothesis
import pytest
from hypothesis import given, assume
from hypothesis.strategies import lists, text, integers, tuples

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

    # Test that we can iterate over dataset multiple times.
    assert list(dataset) == [expected]
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


@hypothesis.settings(deadline=timedelta(seconds=5))
@given(lists(text()))
def test_dataset_iterates_after_pickle(keys: List[str]):
    client = _create_mock_client_with_dummy_objects(TEST_BUCKET, keys)
    dataset = S3IterableDataset.from_bucket(
        TEST_BUCKET,
        client=client,
    )
    assert isinstance(dataset, S3IterableDataset)
    expected = list(dataset)

    actual = []
    iterable = iter(dataset)
    try:
        while True:
            actual.append(next(iterable))
            # Make sure saving/loading state actually keeps the iterator at the same state!
            iterable = pickle.loads(pickle.dumps(iterable))
    except StopIteration:
        pass
    assert [i.key for i in expected] == [i.key for i in actual]


@given(
    lists(text(), unique=True),
    tuples(
        integers(min_value=0, max_value=40),
        integers(min_value=1, max_value=40),
    ),
)
def test_dataset_filter_function(keys: List[str], worker_tuple):
    assume(worker_tuple[0] != worker_tuple[1])
    keys.sort()
    client = _create_mock_client_with_dummy_objects(TEST_BUCKET, keys)
    dataset = S3IterableDataset.from_bucket(
        TEST_BUCKET,
        client=client,
    )
    assert isinstance(dataset, S3IterableDataset)
    assert [obj.key for obj in dataset] == keys
    dataset._num_workers = num_workers = max(worker_tuple)
    dataset._worker_id = worker_id = min(worker_tuple)
    assert [obj.key for obj in dataset] == keys[worker_id::num_workers]


def _verify_dataset(
    dataset: S3IterableDataset,
    expected_keys: Sequence[str],
    expected_count: int,
    object_info_check: Callable[[S3Object], bool],
    *,
    times_to_verify: int = 2,
):
    for _ in range(times_to_verify):
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
