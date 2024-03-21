#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
import logging
import pytest

from hypothesis import given
from hypothesis.strategies import lists, text, integers, floats
from unittest.mock import MagicMock

from s3torchconnectorclient._mountpoint_s3_client import S3Exception

from s3torchconnector._user_agent import UserAgent
from s3torchconnector._version import __version__
from s3torchconnector._s3client import S3Client, MockS3Client, S3ClientConfig

TEST_BUCKET = "test-bucket"
TEST_KEY = "test-key"
TEST_REGION = "us-east-1"
S3_URI = f"s3://{TEST_BUCKET}/{TEST_KEY}"


@pytest.fixture
def s3_client() -> S3Client:
    client = MockS3Client(TEST_REGION, TEST_BUCKET)
    client.add_object(TEST_KEY, b"data")
    return client


def test_get_object_log(s3_client: S3Client, caplog):
    with caplog.at_level(logging.DEBUG):
        s3_client.get_object(TEST_BUCKET, TEST_KEY)
    assert f"GetObject {S3_URI}, object_info is None=True" in caplog.messages


def test_get_object_log_with_info(s3_client: S3Client, caplog):
    with caplog.at_level(logging.DEBUG):
        s3_client.get_object(TEST_BUCKET, TEST_KEY, object_info=MagicMock())
    assert f"GetObject {S3_URI}, object_info is None=False" in caplog.messages


def test_head_object_log(s3_client: S3Client, caplog):
    with caplog.at_level(logging.DEBUG):
        s3_client.head_object(TEST_BUCKET, TEST_KEY)
    assert f"HeadObject {S3_URI}" in caplog.messages


def test_put_object_log(s3_client: S3Client, caplog):
    with caplog.at_level(logging.DEBUG):
        s3_client.put_object(TEST_BUCKET, TEST_KEY)
    assert f"PutObject {S3_URI}" in caplog.messages


def test_list_objects_log(s3_client: S3Client, caplog):
    with caplog.at_level(logging.DEBUG):
        s3_client.list_objects(TEST_BUCKET, TEST_KEY)
    assert f"ListObjects {S3_URI}" in caplog.messages


def test_s3_client_default_user_agent():
    s3_client = S3Client(region=TEST_REGION)
    expected_user_agent = f"s3torchconnector/{__version__}"
    assert s3_client.user_agent_prefix == expected_user_agent
    assert s3_client._client.user_agent_prefix == expected_user_agent


def test_s3_client_custom_user_agent():
    s3_client = S3Client(
        region=TEST_REGION, user_agent=UserAgent(["component/version", "metadata"])
    )
    expected_user_agent = (
        f"s3torchconnector/{__version__} (component/version; metadata)"
    )
    assert s3_client.user_agent_prefix == expected_user_agent
    assert s3_client._client.user_agent_prefix == expected_user_agent


@given(lists(text()))
def test_user_agent_always_starts_with_package_version(comments):
    s3_client = S3Client(region=TEST_REGION, user_agent=UserAgent(comments))
    expected_prefix = f"s3torchconnector/{__version__}"
    assert s3_client.user_agent_prefix.startswith(expected_prefix)
    assert s3_client._client.user_agent_prefix.startswith(expected_prefix)
    comments_str = "; ".join(filter(None, comments))
    if comments_str:
        assert comments_str in s3_client.user_agent_prefix
        assert comments_str in s3_client._client.user_agent_prefix


@given(
    part_size=integers(min_value=5 * 1024, max_value=5 * 1024 * 1024),
    throughput_target_gbps=floats(min_value=10.0, max_value=100.0),
)
def test_s3_client_custom_config(part_size: int, throughput_target_gbps: float):
    # Part size must have values between 5MiB and 5GiB
    part_size = part_size * 1024
    s3_client = S3Client(
        region=TEST_REGION,
        s3client_config=S3ClientConfig(
            part_size=part_size,
            throughput_target_gbps=throughput_target_gbps,
        ),
    )
    assert s3_client._client.part_size == part_size
    assert abs(s3_client._client.throughput_target_gbps - throughput_target_gbps) < 1e-9


def test_s3_client_invalid_part_size_config():
    with pytest.raises(
        S3Exception,
        match="invalid configuration: part size must be at between 5MiB and 5GiB",
    ):
        s3_client = S3Client(
            region=TEST_REGION,
            s3client_config=S3ClientConfig(part_size=1),
        )
        # The client is lazily initialized
        assert s3_client._client.part_size is not None
