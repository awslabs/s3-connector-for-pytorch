import logging
from typing import Iterable, Union

import pytest

from _pytest.outcomes import fail

from s3dataset._s3dataset import (
    S3DatasetException,
    MockMountpointS3Client,
    MountpointS3Client,
)

from s3dataset.s3dataset_base import S3DatasetBase, _parse_s3_uri

logging.basicConfig(
    format="%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s"
)
logging.getLogger().setLevel(1)

log = logging.getLogger(__name__)

TEST_BUCKET = "test-bucket"
TEST_KEY = "test-key"
TEST_REGION = "us-east-1"
S3_PREFIX = "s3://"

@pytest.mark.parametrize(
    "region, client",
    [(TEST_REGION, None), (None, MockMountpointS3Client(TEST_REGION, TEST_BUCKET))],
)
def test_s3dataset_base_validate_arguments_success(region, client):
    try:
        S3DatasetBase._validate_arguments(region, client)
    except ValueError:
        fail("Arguments are valid, check should have passed.")


@pytest.mark.parametrize(
    "region, client, error_msg",
    [
        (None, None, "Either region or client must be valid."),
        ("", None, "Either region or client must be valid."),
        (
            TEST_REGION,
            MockMountpointS3Client(TEST_REGION, TEST_BUCKET),
            "Only one of region / client should be passed.",
        ),
    ],
)
def test_s3dataset_base_validate_arguments_fail(region, client, error_msg):
    with pytest.raises(ValueError) as error:
        S3DatasetBase._validate_arguments(region, client)
    assert str(error.value) == error_msg


@pytest.mark.parametrize(
    "uri, expected_bucket, expected_key",
    [
        (f"{S3_PREFIX}bucket/key", "bucket", "key"),
        (f"{S3_PREFIX}bucket", "bucket", ""),
        (f"{S3_PREFIX}bucket/key/inner-key", "bucket", "key/inner-key"),
    ],
)
def test_s3dataset_base_parse_s3_uri_success(uri, expected_bucket, expected_key):
    bucket, key = _parse_s3_uri(uri)
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
    with pytest.raises(ValueError) as error:
        _parse_s3_uri(uri)
    assert str(error.value) == error_msg


@pytest.mark.parametrize(
    "keys, expected_index",
    [([], 0), (["obj1", "obj2", "obj3", "test"], 3)],
)
def test_objects_to_s3objects(keys: Iterable[str], expected_index: int):

    mock_client = _create_mock_client_with_dummy_objects(TEST_BUCKET, keys)
    bucket_key_pairs = [(TEST_BUCKET, key) for key in keys]
    objects = S3DatasetBase._bucket_keys_to_s3objects(mock_client, bucket_key_pairs)
    index = 0
    for index, object in enumerate(objects):
        assert object is not None
        assert object.bucket == TEST_BUCKET
        assert object.key == keys[index]
        assert object.object_info == None
        assert object.stream is not None
    assert index == expected_index


@pytest.mark.parametrize(
    "prefix, keys, expected_index",
    [
        (None, ["obj1", "obj2", "obj3", "test", "test2"], 4),
        ("", ["obj1", "obj2", "obj3", "test", "test2"], 4),
        ("obj", ["obj1", "obj2", "obj3", "test", "test2"], 2),
    ],
)
def test_list_objects_for_bucket(prefix: str, keys: Iterable[str], expected_index: int):
    mock_client = _create_mock_client_with_dummy_objects(TEST_BUCKET, keys)
    objects = S3DatasetBase._list_objects_for_bucket(mock_client, TEST_BUCKET, prefix)
    index = 0
    for index, object in enumerate(objects):
        assert object is not None
        assert object.bucket == TEST_BUCKET
        assert object.key == keys[index]
        assert object.object_info is not None
        assert object.object_info.key == keys[index]
        assert object.stream is not None
    assert index == expected_index


@pytest.mark.parametrize(
    "prefix, keys, error_msg",
    [
        (
            None,
            ["obj1", "obj2", "obj3", "test", "test2"],
            "Service error: The bucket does not exist",
        ),
        (
            "",
            ["obj1", "obj2", "obj3", "test", "test2"],
            "Service error: The bucket does not exist",
        ),
        (
            "obj",
            ["obj1", "obj2", "obj3", "test", "test2"],
            "Service error: The bucket does not exist",
        ),
    ],
)
def test_list_objects_for_bucket_invalid(
    prefix: str, keys: Iterable[str], error_msg: str
):
    mock_client = _create_mock_client_with_dummy_objects(TEST_BUCKET, keys)
    with pytest.raises(S3DatasetException) as error:
        objects = S3DatasetBase._list_objects_for_bucket(
            mock_client, "DIFFERENT_BUCKET", prefix
        )
        for object in objects:
            assert object is not None
    assert str(error.value) == error_msg


@pytest.mark.parametrize(
    "object_uris, region, client, error_msg",
    [
        (None, "", None, "Either region or client must be valid."),
        ("", None, None, "Either region or client must be valid."),
        ("obj", "", None, "Either region or client must be valid."),
        (
            "obj",
            "us-east-1",
            MountpointS3Client(TEST_REGION),
            "Only one of region / client should be passed.",
        ),
    ],
)
def test_dataset_creation_from_objects_invalid(
    object_uris: Union[str, Iterable[str]],
    region: str,
    client: MountpointS3Client,
    error_msg: str,
):
    with pytest.raises(ValueError) as error:
        S3DatasetBase.from_objects(object_uris, region=region, client=client)
    assert str(error.value) == error_msg


@pytest.mark.parametrize(
    "keys",
    [
        ([]),
        ("single_object"),
        (["obj1", "obj2", "test"]),
    ],
)
def test_dataset_creation_from_objects_with_client(keys: Union[str, Iterable[str]]):
    object_uris = [f"{S3_PREFIX}{TEST_BUCKET}/{key}" for key in keys]
    client = _create_mock_client_with_dummy_objects(TEST_BUCKET, object_uris)
    dataset = S3DatasetBase.from_objects(object_uris, client=client)
    assert dataset is not None


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
    dataset = S3DatasetBase.from_objects(object_uris, region=TEST_REGION)
    assert dataset is not None
    assert dataset.region == TEST_REGION


@pytest.mark.parametrize(
    "keys",
    [
        ([]),
        ("single_object"),
        (["obj1", "obj2", "test"]),
    ],
)
def test_dataset_creation_from_bucket_with_client(keys: Union[str, Iterable[str]]):
    object_uris = [f"{S3_PREFIX}{TEST_BUCKET}/{key}" for key in keys]
    client = _create_mock_client_with_dummy_objects(TEST_BUCKET, object_uris)
    dataset = S3DatasetBase.from_bucket(TEST_BUCKET, client=client)
    assert dataset is not None


def test_dataset_creation_from_bucket_with_region():
    dataset = S3DatasetBase.from_bucket(TEST_BUCKET, region=TEST_REGION)
    assert dataset is not None
    assert dataset.region == TEST_REGION


def _create_mock_client_with_dummy_objects(
    bucket: str, keys: Union[str, Iterable[str]]
):
    mock_client = MockMountpointS3Client(TEST_REGION, bucket)
    for key in keys:
        content = f"{bucket}-{key}-dummyData".encode("utf-8")
        mock_client.add_object(key, bytearray(content))
    return mock_client.create_mocked_client()
