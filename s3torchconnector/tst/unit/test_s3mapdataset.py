#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
import logging
from io import SEEK_END
from typing import Sequence, Callable, Any
from unittest.mock import patch

import pytest

from s3torchconnector import S3MapDataset, S3Reader, S3ReaderConstructor
from s3torchconnector.s3reader import (
    S3ReaderConstructorProtocol,
)
from s3torchconnector._s3client import MockS3Client

from .test_s3dataset_common import (
    TEST_BUCKET,
    TEST_REGION,
    _create_mock_client_with_dummy_objects,
    S3_PREFIX,
    TEST_ENDPOINT,
    READER_TYPE_STRING_TO_CLASS,
)


@pytest.fixture(
    params=[
        S3ReaderConstructor.sequential(),  # Sequential Reader
        S3ReaderConstructor.range_based(),  # Default range-based reader, with buffer
        S3ReaderConstructor.range_based(buffer_size=0),  # range-based reader, no buffer
    ],
    scope="module",
)
def reader_constructor(request) -> S3ReaderConstructor:
    """Provide reader constructor (partial(S3Reader)) instances for all supported reader types."""
    return request.param


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


def test_default_reader_type_from_prefix():
    """Test that SEQUENTIAL is the default reader type when creating from prefix"""
    dataset = S3MapDataset.from_prefix(S3_PREFIX, region=TEST_REGION)
    reader_type_string = S3ReaderConstructor.get_reader_type_string(
        dataset._reader_constructor
    )
    assert reader_type_string == "sequential"


def test_default_reader_type_from_objects():
    """Test that SEQUENTIAL is the default reader type when creating from objects"""
    dataset = S3MapDataset.from_objects([], region=TEST_REGION)
    reader_type_string = S3ReaderConstructor.get_reader_type_string(
        dataset._reader_constructor
    )
    assert reader_type_string == "sequential"


def test_dataset_creation_from_prefix_with_reader_constructor(
    reader_constructor: S3ReaderConstructorProtocol,
):
    dataset = S3MapDataset.from_prefix(
        S3_PREFIX,
        region=TEST_REGION,
        reader_constructor=reader_constructor,
    )
    assert isinstance(dataset, S3MapDataset)
    assert dataset.region == TEST_REGION
    assert dataset._reader_constructor == reader_constructor


def test_dataset_creation_from_objects_with_reader_constructor(
    reader_constructor: S3ReaderConstructorProtocol,
):
    dataset = S3MapDataset.from_objects(
        [], region=TEST_REGION, reader_constructor=reader_constructor
    )
    assert isinstance(dataset, S3MapDataset)
    assert dataset.region == TEST_REGION
    assert dataset._reader_constructor == reader_constructor


@pytest.mark.parametrize(
    "keys",
    [
        ["obj1"],
        ["obj1", "obj2", "obj3"],
        ["obj1", "obj2", "obj3", "test"],
    ],
)
def test_dataset_creation_from_objects(
    keys: Sequence[str], reader_constructor: S3ReaderConstructorProtocol
):
    object_uris = [f"{S3_PREFIX}/{key}" for key in keys]
    dataset = S3MapDataset.from_objects(
        object_uris, region=TEST_REGION, reader_constructor=reader_constructor
    )

    # use mock client for unit testing
    client = _create_mock_client_with_dummy_objects(TEST_BUCKET, keys)
    dataset._client = client

    assert isinstance(dataset, S3MapDataset)
    assert len(dataset) == len(keys)
    for index, key in enumerate(keys):
        verify_item(dataset, index, key, reader_constructor)


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
    reader_constructor: S3ReaderConstructorProtocol,
):
    dataset = S3MapDataset.from_prefix(
        s3_uri=prefix, region=TEST_REGION, reader_constructor=reader_constructor
    )
    client = _create_mock_client_with_dummy_objects(TEST_BUCKET, keys)
    dataset._client = client
    assert isinstance(dataset, S3MapDataset)
    assert len(dataset) == len(expected_keys)

    for index, key in enumerate(expected_keys):
        verify_item(dataset, index, key, reader_constructor)


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
    dataset = S3MapDataset.from_prefix(
        s3_uri=S3_PREFIX,
        region=TEST_REGION,
        transform=transform,
        reader_constructor=reader_constructor,
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
    key: str,
    transform: Callable[[S3Reader], Any],
    expected: Any,
    reader_constructor: S3ReaderConstructorProtocol,
):
    object_uris = f"{S3_PREFIX}/{key}"

    dataset = S3MapDataset.from_objects(
        object_uris,
        region=TEST_REGION,
        transform=transform,
        reader_constructor=reader_constructor,
    )

    # use mock client for unit testing
    client = _create_mock_client_with_dummy_objects(TEST_BUCKET, [key])
    dataset._client = client

    assert isinstance(dataset, S3MapDataset)
    assert list(dataset) == [expected]


def test_from_prefix_seek_no_head(reader_constructor: S3ReaderConstructorProtocol):
    dataset = S3MapDataset.from_prefix(
        S3_PREFIX, region=TEST_REGION, reader_constructor=reader_constructor
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


def test_user_agent_includes_dataset_and_reader_type(
    reader_constructor: S3ReaderConstructorProtocol,
):
    """Test that user agent includes dataset type and reader type."""
    dataset = S3MapDataset.from_prefix(
        S3_PREFIX, region=TEST_REGION, reader_constructor=reader_constructor
    )
    dataset._get_client()

    user_agent = dataset._client.user_agent_prefix

    reader_type_string = S3ReaderConstructor.get_reader_type_string(reader_constructor)
    # expect: sequential / range_based

    assert "md/dataset#map" in user_agent
    assert f"md/reader_type#{reader_type_string}" in user_agent


def verify_item(
    dataset: S3MapDataset,
    index: int,
    expected_key: str,
    reader_constructor: S3ReaderConstructorProtocol,
):
    data = dataset[index]

    assert data is not None
    assert data.bucket == TEST_BUCKET
    assert data.key == expected_key
    reader_type_string = S3ReaderConstructor.get_reader_type_string(reader_constructor)
    assert isinstance(
        data,
        READER_TYPE_STRING_TO_CLASS[reader_type_string],
    )
    expected_content = f"{TEST_BUCKET}-{expected_key}-dummyData".encode()
    content = data.read()
    assert content == expected_content
