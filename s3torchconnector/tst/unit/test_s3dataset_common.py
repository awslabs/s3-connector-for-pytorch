#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import logging
from typing import Iterable, Union, Sequence

from unittest.mock import patch, MagicMock
import pytest

from s3torchconnector import S3Exception
from s3torchconnector._s3client import MockS3Client

from s3torchconnector._s3dataset_common import (
    parse_s3_uri,
    get_objects_from_prefix,
    get_objects_from_uris,
    get_shard_index,
    get_shards_count,
)

logging.basicConfig(
    format="%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s"
)
logging.getLogger().setLevel(1)

log = logging.getLogger(__name__)

TEST_BUCKET = "test-bucket"
TEST_KEY = "test-key"
TEST_REGION = "us-east-1"
S3_PREFIX = f"s3://{TEST_BUCKET}"
TEST_ENDPOINT = "https://s3.us-east-1.amazonaws.com"


@pytest.mark.parametrize(
    "uri, expected_bucket, expected_key",
    [
        (f"s3://bucket/key", "bucket", "key"),
        (f"s3://bucket", "bucket", ""),
        (f"s3://bucket/key/inner-key", "bucket", "key/inner-key"),
    ],
)
def test_s3dataset_base_parse_s3_uri_success(uri, expected_bucket, expected_key):
    bucket, key = parse_s3_uri(uri)
    assert bucket == expected_bucket
    assert key == expected_key


@pytest.mark.parametrize(
    "uri, error_msg",
    [
        (None, "Only s3:// URIs are supported"),
        ("", "Only s3:// URIs are supported"),
        ("s3a://bucket/key", "Only s3:// URIs are supported"),
        ("s3://", "Bucket name must be non-empty"),
        ("s3:///key", "Bucket name must be non-empty"),
    ],
)
def test_s3dataset_base_parse_s3_uri_fail(uri, error_msg):
    with pytest.raises(ValueError, match=f"^{error_msg}$"):
        parse_s3_uri(uri)


@pytest.mark.parametrize(
    "prefix, keys, expected_count",
    [
        ("", ["obj1", "obj2", "obj3", "test", "test2"], 5),
        ("obj", ["obj1", "obj2", "obj3", "test", "test2"], 3),
    ],
)
def test_get_objects_from_prefix(prefix: str, keys: Sequence[str], expected_count: int):
    mock_client = _create_mock_client_with_dummy_objects(TEST_BUCKET, keys)
    bucket_key_pairs = get_objects_from_prefix(f"{S3_PREFIX}/{prefix}", mock_client)
    count = 0
    for index, bucket_key_pair in enumerate(bucket_key_pairs):
        count += 1
        assert bucket_key_pair is not None
        assert bucket_key_pair.bucket == TEST_BUCKET
        assert bucket_key_pair.key == keys[index]
    assert count == expected_count


def test_list_objects_for_bucket_invalid():
    mock_client = _create_mock_client_with_dummy_objects(TEST_BUCKET, [])
    with pytest.raises(S3Exception, match="Service error: The bucket does not exist"):
        objects = get_objects_from_prefix(
            "s3://DIFFERENT_BUCKET",
            mock_client,
        )
        next(iter(objects))


@pytest.mark.parametrize(
    "object_uris, expected_keys",
    [([], []), ([f"{S3_PREFIX}/obj1", f"{S3_PREFIX}/obj2"], ["obj1", "obj2"])],
)
def test_get_objects_from_uris_success(
    object_uris: Sequence[str], expected_keys: Sequence[str]
):
    mock_client = _create_mock_client_with_dummy_objects(TEST_BUCKET, expected_keys)
    bucket_key_pairs = get_objects_from_uris(object_uris, mock_client)
    count = 0
    for index, bucket_key_pair in enumerate(bucket_key_pairs):
        count += 1
        assert bucket_key_pair is not None
        assert bucket_key_pair.bucket == TEST_BUCKET
        assert bucket_key_pair.key == expected_keys[index]
    assert count == len(expected_keys)


@pytest.mark.parametrize(
    "uri, error_msg",
    [
        ("", "Only s3:// URIs are supported"),
        ("s3a://bucket/key", "Only s3:// URIs are supported"),
        ("s3://", "Bucket name must be non-empty"),
        ("s3:///key", "Bucket name must be non-empty"),
    ],
)
def test_get_objects_from_uris_fail(uri, error_msg):
    mock_client = _create_mock_client_with_dummy_objects(TEST_BUCKET, [])
    with pytest.raises(ValueError, match=f"^{error_msg}$"):
        get_objects_from_uris(uri, mock_client)


@pytest.mark.parametrize(
    "rank, worker_id, num_workers, expected_shard_index",
    [
        (0, 0, 1, 0),
        (0, 1, 2, 1),
        (1, 0, 2, 2),
        (2, 1, 3, 7),
        (5, 1, 10, 51),
    ],
)
@patch("torch.utils.data.get_worker_info")
def test_get_shard_index(
        get_worker_info_mock,
        rank: int,
        worker_id: int,
        num_workers: int,
        expected_shard_index: int
):
    worker_info_mock = MagicMock(id=worker_id, num_workers=num_workers)
    get_worker_info_mock.return_value = worker_info_mock

    shard_index = get_shard_index(rank)
    assert shard_index == expected_shard_index


@pytest.mark.parametrize(
    "world_size, worker_id, num_workers, expected_shards_count",
    [
        (1, 0, 1, 1),
        (0, 1, 2, 2),
        (1, 0, 2, 2),
        (2, 1, 3, 6),
        (5, 1, 10, 50),
    ],
)
@patch("torch.utils.data.get_worker_info")
def test_get_shards_count(
        get_worker_info_mock,
        world_size: int,
        worker_id: int,
        num_workers: int,
        expected_shards_count: int
):
    worker_info_mock = MagicMock(id=worker_id, num_workers=num_workers)
    get_worker_info_mock.return_value = worker_info_mock

    shards_count = get_shards_count(world_size)
    assert shards_count == expected_shards_count

def _create_mock_client_with_dummy_objects(
    bucket: str, keys: Union[str, Iterable[str]]
):
    mock_client = MockS3Client(TEST_REGION, bucket)
    for key in keys:
        content = f"{bucket}-{key}-dummyData".encode()
        mock_client.add_object(key, content)
    return mock_client
