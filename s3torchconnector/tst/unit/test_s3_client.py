#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
import logging
import pytest
import platform

from hypothesis import given, example
from hypothesis.strategies import lists, text, integers, floats
from unittest.mock import MagicMock

from s3torchconnectorclient._mountpoint_s3_client import S3Exception

from s3torchconnector._user_agent import UserAgent
from s3torchconnector._version import __version__
from s3torchconnector._s3client import S3Client, MockS3Client, S3ClientConfig

DEFAULT_USER_AGENT = UserAgent.get_default_prefix()

TEST_BUCKET = "test-bucket"
TEST_KEY = "test-key"
TEST_REGION = "us-east-1"
S3_URI = f"s3://{TEST_BUCKET}/{TEST_KEY}"

KiB = 1 << 10
MiB = 1 << 20
GiB = 1 << 30


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


def test_delete_object_log(s3_client: S3Client, caplog):
    with caplog.at_level(logging.DEBUG):
        s3_client.delete_object(TEST_BUCKET, TEST_KEY)
    assert f"DeleteObject {S3_URI}" in caplog.messages


def test_copy_object_log(s3_client: S3Client, caplog):
    dst_bucket, dst_key = "dst_bucket", "dst_key"

    with caplog.at_level(logging.DEBUG):
        s3_client.copy_object(TEST_BUCKET, TEST_KEY, dst_bucket, dst_key)
    assert f"CopyObject {S3_URI} to s3://{dst_bucket}/{dst_key}" in caplog.messages


def test_s3_client_default_user_agent():
    s3_client = S3Client(region=TEST_REGION)
    assert s3_client.user_agent_prefix == DEFAULT_USER_AGENT
    assert s3_client._client.user_agent_prefix == DEFAULT_USER_AGENT


def test_s3_client_custom_user_agent():
    s3_client = S3Client(
        region=TEST_REGION, user_agent=UserAgent(["component/version", "metadata"])
    )
    expected_user_agent = f"{DEFAULT_USER_AGENT} (component/version; metadata)"
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
    part_size=integers(min_value=5 * MiB, max_value=5 * GiB),
    throughput_target_gbps=floats(min_value=10.0, max_value=100.0),
    max_attempts=integers(min_value=1, max_value=10),
)
@example(part_size=5 * MiB, throughput_target_gbps=10.0, max_attempts=1)
@example(part_size=5 * GiB, throughput_target_gbps=15.0, max_attempts=10)
def test_s3_client_custom_config(
    part_size: int, throughput_target_gbps: float, max_attempts: int
):
    # Part size must have values between 5MiB and 5GiB
    s3_client = S3Client(
        region=TEST_REGION,
        s3client_config=S3ClientConfig(
            part_size=part_size,
            throughput_target_gbps=throughput_target_gbps,
            max_attempts=max_attempts,
        ),
    )
    assert s3_client._client.part_size == part_size
    assert s3_client._client.throughput_target_gbps == throughput_target_gbps
    assert s3_client._client.unsigned is False
    assert s3_client._client.max_attempts == max_attempts


@pytest.mark.parametrize(
    "part_size",
    [
        1,
        2 * KiB,
        5 * MiB - 1,
        5 * GiB + 1,
        6 * GiB,
    ],
)
def test_s3_client_invalid_part_size_config(part_size: int):
    with pytest.raises(
        S3Exception,
        match="invalid configuration: part size must be at between 5MiB and 5GiB",
    ):
        s3_client = S3Client(
            region=TEST_REGION,
            s3client_config=S3ClientConfig(part_size=part_size),
        )
        # The client is lazily initialized
        assert s3_client._client.part_size == part_size


def test_unsigned_s3_client():
    s3_client = S3Client(
        region=TEST_REGION,
        s3client_config=S3ClientConfig(unsigned=True),
    )
    assert s3_client._client.unsigned is True


def test_force_path_style_s3_client():
    s3_client = S3Client(
        region=TEST_REGION,
        s3client_config=S3ClientConfig(force_path_style=True),
    )
    assert s3_client._client.force_path_style is True


def test_s3_client_different_configs():
    s3_client_slow = S3Client(
        region=TEST_REGION,
        s3client_config=S3ClientConfig(throughput_target_gbps=5, part_size=5 * MiB),
    )
    s3_client_fast = S3Client(
        region=TEST_REGION,
        s3client_config=S3ClientConfig(throughput_target_gbps=500, part_size=32 * MiB),
    )
    assert s3_client_slow._client.throughput_target_gbps == 5
    assert s3_client_slow._client.part_size == 5 * MiB
    assert s3_client_fast._client.throughput_target_gbps == 500
    assert s3_client_fast._client.part_size == 32 * MiB
