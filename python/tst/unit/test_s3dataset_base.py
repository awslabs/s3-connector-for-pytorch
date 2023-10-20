import logging

import pytest

from _pytest.outcomes import fail
from s3dataset._s3dataset import (
    S3DatasetException,
    GetObjectStream,
    ListObjectStream,
    ObjectInfo,
    MockMountpointS3Client,
)

from python.src.s3dataset.s3dataset_base import S3DatasetBase, _parse_s3_uri

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
    [
        (TEST_REGION, None),
        (None, MockMountpointS3Client(TEST_REGION, TEST_BUCKET))
    ],
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
        (TEST_REGION, MockMountpointS3Client(TEST_REGION, TEST_BUCKET), "Only one of region / client should be passed.")
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