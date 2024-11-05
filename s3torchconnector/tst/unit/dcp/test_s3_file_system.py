#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import pytest

from s3torchconnector._s3client import MockS3Client
from s3torchconnector.dcp.s3_file_system import S3FileSystem
from s3torchconnectorclient import S3Exception

TEST_REGION = "eu-east-1"
TEST_BUCKET = "test-bucket"
TEST_KEY = "test-key.txt"
TEST_DATA = b"test data\n"

_build_s3_uri = lambda bucket, key: f"s3://{bucket}/{key}"


# TODO: transform all "path"s to actual S3 URIs.


def test_create_stream():
    mock_client = MockS3Client(TEST_REGION, TEST_BUCKET)
    s3fs = S3FileSystem(TEST_REGION, mock_client)

    path = _build_s3_uri(TEST_BUCKET, TEST_KEY)

    with s3fs.create_stream(path, "wb") as writer:
        writer.write(TEST_DATA)

    assert s3fs.exists(path)

    with s3fs.create_stream(path, "rb") as reader:
        assert reader.read() == TEST_DATA


def test_create_stream_raise_when_mode_is_unknown():
    mock_client = MockS3Client(TEST_REGION, TEST_BUCKET)
    s3fs = S3FileSystem(TEST_REGION, mock_client)

    path = _build_s3_uri(TEST_BUCKET, TEST_KEY)

    with pytest.raises(ValueError) as excinfo:
        with s3fs.create_stream(path, "foobar"):
            pass

    assert (
        str(excinfo.value)
        == 'Invalid mode=\'foobar\' argument: `create_stream` only supports "rb" (read) or "wb" (write) modes'
    )


def test_concat_path():
    mock_client = MockS3Client(TEST_REGION, TEST_BUCKET)
    s3fs = S3FileSystem(TEST_REGION, mock_client)

    path = _build_s3_uri(TEST_BUCKET, TEST_KEY)
    suffix = "suffix"

    assert s3fs.concat_path(path, suffix) == path + "/" + suffix


def test_init_path():
    mock_client = MockS3Client(TEST_REGION, TEST_BUCKET)
    s3fs = S3FileSystem(TEST_REGION, mock_client)

    path = "s3://bucket/key"
    assert s3fs.init_path(path) == path


def test_rename():
    src_key, src_data = ("src_key.txt", b"src data\n")
    dst_key = "dst_key.txt"

    mock_client = MockS3Client(TEST_REGION, TEST_BUCKET)
    mock_client.add_object(src_key, src_data)
    s3fs = S3FileSystem(TEST_REGION, mock_client)

    src_path = _build_s3_uri(TEST_BUCKET, src_key)
    dst_path = _build_s3_uri(TEST_BUCKET, dst_key)
    s3fs.rename(src_path, dst_path)

    assert s3fs.exists(src_path) is False
    assert s3fs.exists(dst_path) is True


def test_rename_raises_when_buckets_are_different():
    src_key, src_data = ("src_key.txt", b"src data\n")
    dst_key = "dst_key"

    mock_client = MockS3Client(TEST_REGION, TEST_BUCKET)
    mock_client.add_object(src_key, src_data)
    s3fs = S3FileSystem(TEST_REGION, mock_client)

    src_path = _build_s3_uri(TEST_BUCKET, src_key)
    dst_path = _build_s3_uri(TEST_BUCKET + "-another", dst_key)

    with pytest.raises(ValueError) as excinfo:
        s3fs.rename(src_path, dst_path)

    assert (
        str(excinfo.value)
        == "Source and destination buckets cannot be different (`rename` does not support cross-buckets operations)"
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

    assert s3fs.exists(_build_s3_uri(TEST_BUCKET, TEST_KEY)) is True


def test_exists_false_when_key_does_not_exist():
    path = _build_s3_uri(TEST_BUCKET, "test_key_does_not_exist")

    mock_client = MockS3Client(TEST_REGION, TEST_BUCKET)
    s3fs = S3FileSystem(TEST_REGION, mock_client)

    assert s3fs.exists(path) is False


@pytest.mark.parametrize(
    "exception", [S3Exception("Some S3 exception"), ValueError("Some random error")]
)
def test_exists_raise_when_underlying_exception_is_unexpected(exception):
    path = _build_s3_uri(TEST_BUCKET, "test_key_does_not_exist")

    mock_client = MockS3Client(TEST_REGION, TEST_BUCKET)

    def raise_s3_exception(_, __):
        raise exception

    mock_client.head_object = raise_s3_exception
    s3fs = S3FileSystem(TEST_REGION, mock_client)

    with pytest.raises(Exception) as excinfo:
        s3fs.exists(path)

    assert excinfo.type == type(exception)
    assert str(excinfo.value) == str(exception)


def test_rm_file():
    mock_client = MockS3Client(TEST_REGION, TEST_BUCKET)
    mock_client.add_object(TEST_KEY, TEST_DATA)
    s3fs = S3FileSystem(TEST_REGION, mock_client)

    path = _build_s3_uri(TEST_BUCKET, TEST_KEY)

    assert s3fs.exists(path) is True

    s3fs.rm_file(path)

    assert s3fs.exists(path) is False


def test_validate_checkpoint_id():
    path = _build_s3_uri(TEST_BUCKET, TEST_KEY)
    assert S3FileSystem.validate_checkpoint_id(path) is True


@pytest.mark.parametrize("checkpoint_id", ["foobar", "s3:///"])
def test_validate_checkpoint_id_returns_false_when_path_is_invalid(checkpoint_id):
    assert S3FileSystem.validate_checkpoint_id(checkpoint_id) is False
