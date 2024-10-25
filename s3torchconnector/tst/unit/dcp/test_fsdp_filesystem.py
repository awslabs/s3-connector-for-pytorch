#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import os
import time
from pathlib import Path
from typing import Union

import pytest

from s3torchconnector._s3client import MockS3Client
from s3torchconnector.dcp.fsdp_filesystem import S3FileSystem
from s3torchconnectorclient import S3Exception

TEST_REGION = "eu-east-1"
TEST_BUCKET = "test-bucket"
TEST_KEY = "test-key.txt"
TEST_DATA = b"test data\n"
TEST_PATH = f"s3://{TEST_BUCKET}/{TEST_KEY}"


def test_create_stream():
    mock_client = MockS3Client(TEST_REGION, TEST_BUCKET)
    s3fs = S3FileSystem(TEST_REGION, mock_client)

    with s3fs.create_stream(TEST_PATH, "wb") as writer:
        writer.write(TEST_DATA)

    assert s3fs.exists(TEST_PATH)

    with s3fs.create_stream(TEST_PATH, "rb") as reader:
        assert reader.read() == TEST_DATA


def test_create_stream_raise_when_mode_is_unknown():
    mock_client = MockS3Client(TEST_REGION, TEST_BUCKET)
    s3fs = S3FileSystem(TEST_REGION, mock_client)

    with pytest.raises(ValueError) as excinfo:
        with s3fs.create_stream(TEST_PATH, "foobar"):
            pass

    assert (
        str(excinfo.value)
        == "Invalid mode='foobar' mode argument: create_stream only supports rb (read mode) & wb (write mode)"
    )


@pytest.mark.parametrize(
    "path,suffix,expected",
    [
        ("str_path", "suffix", "str_path/suffix"),
        (Path("path_path"), "suffix", "path_path/suffix"),
    ],
)
def test_concat_path(path: Union[str, os.PathLike], suffix: str, expected: str):
    mock_client = MockS3Client(TEST_REGION, TEST_BUCKET)
    s3fs = S3FileSystem(TEST_REGION, mock_client)

    assert s3fs.concat_path(path, suffix) == expected


@pytest.mark.parametrize("path", ["str_path", Path("path_path")])
def test_init_path(path: Union[str, os.PathLike]):
    mock_client = MockS3Client(TEST_REGION, TEST_BUCKET)
    s3fs = S3FileSystem(TEST_REGION, mock_client)

    assert s3fs.init_path(path) == path


def test_rename():
    src_key, src_data = ("src_key.txt", b"src data\n")
    dst_key = "dst_key.txt"
    src_path = _build_s3_uri(TEST_BUCKET, src_key)
    dst_path = _build_s3_uri(TEST_BUCKET, dst_key)

    mock_client = MockS3Client(TEST_REGION, TEST_BUCKET)
    mock_client.add_object(src_key, src_data)
    s3fs = S3FileSystem(TEST_REGION, mock_client)

    s3fs.rename(src_path, dst_path)

    assert s3fs.exists(src_path) is False
    assert s3fs.exists(dst_path) is True


def test_rename_raises_when_buckets_are_different():
    src_key, src_data = ("src_key.txt", b"src data\n")
    dst_key = "dst_key"
    src_path = _build_s3_uri(TEST_BUCKET, src_key)
    dst_path = _build_s3_uri(TEST_BUCKET + "-another", dst_key)

    mock_client = MockS3Client(TEST_REGION, TEST_BUCKET)
    mock_client.add_object(src_key, src_data)
    s3fs = S3FileSystem(TEST_REGION, mock_client)

    with pytest.raises(ValueError) as excinfo:
        s3fs.rename(src_path, dst_path)

    assert (
        str(excinfo.value)
        == "Source and destination buckets cannot be different (rename does not support cross-buckets operations)"
    )


def test_rename_raises_when_retries_fail(caplog):
    src_key, src_data = ("src_key.txt", b"src data\n")
    dst_key = "dst_key"
    src_path = _build_s3_uri(TEST_BUCKET, src_key)
    dst_path = _build_s3_uri(TEST_BUCKET, dst_key)

    mock_client = MockS3Client(TEST_REGION, TEST_BUCKET)
    mock_client.add_object(src_key, src_data)

    def raise_s3_exception(bucket, key):
        raise S3Exception("Custom exception; failed to delete (after retries)")

    mock_client.delete_object = raise_s3_exception
    s3fs = S3FileSystem(TEST_REGION, mock_client)

    with pytest.raises(S3Exception) as excinfo:
        s3fs.rename(src_path, dst_path)

    assert str(excinfo.value) == "Custom exception; failed to delete (after retries)"
    assert "this was the 3rd time calling it" in caplog.text


@pytest.mark.skip(reason="method not implemented (no-op)")
def test_mkdir():
    pass


def test_exists_true_when_key_exists():
    mock_client = MockS3Client(TEST_REGION, TEST_BUCKET)
    mock_client.add_object(TEST_KEY, TEST_DATA)
    s3fs = S3FileSystem(TEST_REGION, mock_client)

    assert s3fs.exists(TEST_PATH) is True


def test_exists_false_when_key_does_not_exist():
    path = _build_s3_uri(TEST_BUCKET, "test_key_does_not_exist")

    mock_client = MockS3Client(TEST_REGION, TEST_BUCKET)
    s3fs = S3FileSystem(TEST_REGION, mock_client)

    assert s3fs.exists(path) is False


def test_rm_file():
    mock_client = MockS3Client(TEST_REGION, TEST_BUCKET)
    mock_client.add_object(TEST_KEY, TEST_DATA)
    s3fs = S3FileSystem(TEST_REGION, mock_client)

    assert s3fs.exists(TEST_PATH) is True

    s3fs.rm_file(TEST_PATH)

    assert s3fs.exists(TEST_PATH) is False


def test_validate_checkpoint_id():
    assert S3FileSystem.validate_checkpoint_id(TEST_PATH) is True


@pytest.mark.parametrize("checkpoint_id", ["foobar", "s3:///"])
def test_validate_checkpoint_id_returns_false_when_path_is_invalid(checkpoint_id):
    assert S3FileSystem.validate_checkpoint_id(checkpoint_id) is False


def _build_s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"
