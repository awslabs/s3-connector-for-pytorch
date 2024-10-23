#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import os
from pathlib import Path
from typing import Union

import pytest

from s3torchconnector._s3client import MockS3Client
from s3torchconnector.dcp.fsdp_filesystem import S3FileSystem

TEST_REGION = "eu-east-2"
TEST_BUCKET = "test_bucket"


def test_init():
    s3fs = S3FileSystem(TEST_REGION)

    assert s3fs.path == ""
    assert s3fs.region == TEST_REGION
    assert s3fs.client is not None
    assert s3fs.checkpoint is not None


@pytest.mark.skip(
    reason="not testable as-is (requires contract changes in `S3Checkpoint`)"
)
def test_create_stream():
    pass


def test_create_stream_raise_when_mode_is_unknown():
    s3fs = S3FileSystem(TEST_REGION)
    mode = "foo"

    with pytest.raises(ValueError) as excinfo:
        with s3fs.create_stream("path", mode):
            pass

    assert (
        str(excinfo.value)
        == "Invalid mode argument, create_stream only supports rb (read mode) & wb (write mode)"
    )


@pytest.mark.parametrize("path", ["str_path", Path("path_path")])
def test_concat_path(path: Union[str, os.PathLike]):
    s3fs = S3FileSystem(TEST_REGION)

    concatenated_path = s3fs.concat_path(path, "suffix")

    if type(path) == str:
        assert concatenated_path == "str_path/suffix"
    else:
        assert concatenated_path == "path_path/suffix"


@pytest.mark.parametrize("path", ["str_path", Path("path_path")])
def test_init_path(path: Union[str, os.PathLike]):
    s3fs = S3FileSystem(TEST_REGION)

    s3fs.init_path(path)

    assert s3fs.path == path


def test_rename():
    src_key, src_data = ("src_key", b"src_data")
    dst_key = "dst_key"
    src_path = _build_s3_uri(TEST_BUCKET, src_key)
    dst_path = _build_s3_uri(TEST_BUCKET, dst_key)

    mock_client = MockS3Client(TEST_REGION, TEST_BUCKET)
    mock_client.add_object(src_key, src_data)

    s3fs = S3FileSystem(TEST_REGION, mock_client)
    s3fs.rename(src_path, dst_path)
    assert s3fs.exists(src_path) is False
    assert s3fs.exists(dst_path) is True


@pytest.mark.skip(reason="method not implemented (no-op)")
def test_mkdir():
    pass


def test_exists_true_when_key_exists():
    key, data = ("test_key_exists", b"test_data")
    path = _build_s3_uri(TEST_BUCKET, key)

    mock_client = MockS3Client(TEST_REGION, TEST_BUCKET)
    mock_client.add_object(key, data)

    s3fs = S3FileSystem(TEST_REGION, mock_client)

    assert s3fs.exists(path) is True


def test_exists_false_when_key_does_not_exist():
    key = "test_key_does_not_exist"
    path = _build_s3_uri(TEST_BUCKET, key)

    mock_client = MockS3Client(TEST_REGION, TEST_BUCKET)

    s3fs = S3FileSystem(TEST_REGION, mock_client)

    assert s3fs.exists(path) is False


def test_rm_file():
    key, data = ("test_rm_file", b"test_data")
    path = _build_s3_uri(TEST_BUCKET, key)

    mock_client = MockS3Client(TEST_REGION, TEST_BUCKET)
    mock_client.add_object(key, data)

    s3fs = S3FileSystem(TEST_REGION, mock_client)
    assert s3fs.exists(path) is True

    s3fs.rm_file(path)
    assert s3fs.exists(path) is False


def _build_s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"
