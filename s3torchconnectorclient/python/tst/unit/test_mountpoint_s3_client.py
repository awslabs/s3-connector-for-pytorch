#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import logging
import pickle
from typing import Set

import pytest
from s3torchconnectorclient._mountpoint_s3_client import (
    S3Exception,
    GetObjectStream,
    ListObjectStream,
    PutObjectStream,
    MockMountpointS3Client,
    MountpointS3Client,
)

from s3torchconnectorclient import LOG_TRACE

logging.basicConfig(
    format="%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s"
)
logging.getLogger().setLevel(LOG_TRACE)


REGION = "us-east-1"
MOCK_BUCKET = "mock-bucket"


@pytest.mark.parametrize(
    "key, data, part_size",
    [
        ("hello_world.txt", b"Hello, world!", 1000),
        ("multipart", b"The quick brown fox jumps over the lazy dog.", 2),
    ],
)
def test_get_object(key: str, data: bytes, part_size: int):
    mock_client = MockMountpointS3Client(REGION, MOCK_BUCKET, part_size=part_size)
    mock_client.add_object(key, data)
    client = mock_client.create_mocked_client()

    stream = client.get_object(MOCK_BUCKET, key)
    _assert_isinstance(stream, GetObjectStream)

    returned_data = b"".join(stream)
    assert returned_data == data


def test_get_object_part_size():
    mock_client = MockMountpointS3Client(REGION, MOCK_BUCKET, part_size=2)
    mock_client.add_object("key", b"1234567890")
    client = mock_client.create_mocked_client()

    stream = client.get_object(MOCK_BUCKET, "key")
    _assert_isinstance(stream, GetObjectStream)

    expected = [b"12", b"34", b"56", b"78", b"90"]

    assert stream.tell() == 0
    for i, actual in enumerate(stream):
        assert actual == expected[i]
        expected_position = (i + 1) * 2
        assert stream.tell() == expected_position


def test_get_object_bad_bucket():
    mock_client = MockMountpointS3Client(REGION, MOCK_BUCKET)
    client = mock_client.create_mocked_client()

    try:
        client.get_object("does_not_exist", "foo")
    except S3Exception as e:
        assert str(e) == "Service error: The bucket does not exist"


def test_get_object_none_bucket():
    mock_client = MockMountpointS3Client(REGION, MOCK_BUCKET)
    client = mock_client.create_mocked_client()

    try:
        client.get_object(None, "foo")
    except TypeError:
        pass
    else:
        raise AssertionError("Should raise TypeError")


def test_get_object_bad_object():
    mock_client = MockMountpointS3Client(REGION, MOCK_BUCKET)
    client = mock_client.create_mocked_client()

    try:
        client.get_object(MOCK_BUCKET, "does_not_exist")
    except S3Exception as e:
        assert str(e) == "Service error: The key does not exist"


def test_get_object_iterates_once():
    mock_client = MockMountpointS3Client(REGION, MOCK_BUCKET)
    mock_client.add_object("key", b"data")
    client = mock_client.create_mocked_client()

    stream = client.get_object(MOCK_BUCKET, "key")
    _assert_isinstance(stream, GetObjectStream)

    returned_data = b"".join(stream)
    assert returned_data == b"data"

    returned_data = b"".join(stream)
    assert returned_data == b""


def test_get_object_throws_stop_iteration():
    mock_client = MockMountpointS3Client(REGION, MOCK_BUCKET)
    mock_client.add_object("key", b"data")
    client = mock_client.create_mocked_client()

    stream = client.get_object(MOCK_BUCKET, "key")
    _assert_isinstance(stream, GetObjectStream)

    for _ in stream:
        pass

    for _ in range(10):
        try:
            next(stream)
        except StopIteration:
            pass
        else:
            raise AssertionError("Should always throw StopIteration after stream ends")


@pytest.mark.parametrize(
    "expected_keys",
    [
        ({"tests"}),
        ({"multiple", "objects"}),
        (set()),
    ],
)
def test_list_objects(expected_keys):
    mock_client = MockMountpointS3Client(REGION, MOCK_BUCKET)
    for key in expected_keys:
        mock_client.add_object(key, b"")
    client = mock_client.create_mocked_client()

    stream = client.list_objects(MOCK_BUCKET)
    assert isinstance(stream, ListObjectStream)

    object_infos = [object_info for page in stream for object_info in page.object_info]
    keys = {object_info.key for object_info in object_infos}
    assert keys == expected_keys


@pytest.mark.parametrize(
    "prefix, keys, expected_keys",
    [
        ("key", {"test", "key1", "key2", "key3"}, {"key1", "key2", "key3"}),
        (
            "prefix",
            {"prefix/obj1", "prefix/obj2", "test"},
            {"prefix/obj1", "prefix/obj2"},
        ),
        ("test", set(), set()),
    ],
)
def test_list_objects_with_prefix(prefix: str, keys: Set[str], expected_keys: Set[str]):
    mock_client = MockMountpointS3Client(REGION, MOCK_BUCKET)
    for key in keys:
        mock_client.add_object(key, b"")
    client = mock_client.create_mocked_client()

    stream = client.list_objects(MOCK_BUCKET, prefix)
    assert isinstance(stream, ListObjectStream)

    object_infos = [object_info for page in stream for object_info in page.object_info]
    assert len(object_infos) == len(expected_keys)
    keys = {object_info.key for object_info in object_infos}
    assert keys == expected_keys


@pytest.mark.parametrize(
    "data_to_write,part_size",
    [
        (b"Hello, world!", 2000),
        (b"MultiPartUpload", 2),
        (b"", 2000),
    ],
)
def test_put_object(data_to_write: bytes, part_size: int):
    mock_client = MockMountpointS3Client(REGION, MOCK_BUCKET, part_size=part_size)
    client = mock_client.create_mocked_client()

    put_stream = client.put_object(MOCK_BUCKET, "key")
    assert isinstance(put_stream, PutObjectStream)

    put_stream.write(data_to_write)
    put_stream.close()

    get_stream = client.get_object(MOCK_BUCKET, "key")
    assert b"".join(get_stream) == data_to_write


def test_put_object_overwrite():
    mock_client = MockMountpointS3Client(REGION, MOCK_BUCKET)
    mock_client.add_object("key", b"before")
    client = mock_client.create_mocked_client()

    put_stream = client.put_object(MOCK_BUCKET, "key")
    assert isinstance(put_stream, PutObjectStream)

    put_stream.write(b"after")
    put_stream.close()

    get_stream = client.get_object(MOCK_BUCKET, "key")
    assert b"".join(get_stream) == b"after"


def test_put_object_no_multiple_close():
    mock_client = MockMountpointS3Client(REGION, MOCK_BUCKET)
    client = mock_client.create_mocked_client()

    put_stream = client.put_object(MOCK_BUCKET, "key")
    assert isinstance(put_stream, PutObjectStream)

    put_stream.write(b"")
    put_stream.close()
    with pytest.raises(S3Exception) as e:
        put_stream.close()
    assert str(e.value) == "Cannot close object more than once"


def test_put_object_no_write_after_close():
    mock_client = MockMountpointS3Client(REGION, MOCK_BUCKET)
    client = mock_client.create_mocked_client()

    put_stream = client.put_object(MOCK_BUCKET, "key")
    assert isinstance(put_stream, PutObjectStream)

    put_stream.write(b"")
    put_stream.close()
    with pytest.raises(S3Exception) as e:
        put_stream.write(b"")
    assert str(e.value) == "Cannot write to closed object"


def test_put_object_with_storage_class():
    mock_client = MockMountpointS3Client(REGION, MOCK_BUCKET)
    client = mock_client.create_mocked_client()

    put_stream = client.put_object(MOCK_BUCKET, "key", "STANDARD_IA")
    assert isinstance(put_stream, PutObjectStream)

    put_stream.write(b"")
    put_stream.close()


# TODO: Add hypothesis setup after aligning on limits
def test_mountpoint_client_pickles():
    expected_profile = None
    expected_no_sign_request = False
    expected_region = REGION
    expected_part_size = 5 * 2**20
    expected_throughput_target_gbps = 3.5

    client = MountpointS3Client(
        region=expected_region,
        user_agent_prefix="unit-tests",
        part_size=expected_part_size,
        throughput_target_gbps=expected_throughput_target_gbps,
        profile=expected_profile,
        no_sign_request=expected_no_sign_request,
    )
    dumped = pickle.dumps(client)
    loaded = pickle.loads(dumped)

    assert isinstance(dumped, bytes)
    assert isinstance(loaded, MountpointS3Client)
    assert client is not loaded

    assert client.region == loaded.region == expected_region
    assert client.part_size == loaded.part_size == expected_part_size
    assert (
        client.throughput_target_gbps
        == loaded.throughput_target_gbps
        == expected_throughput_target_gbps
    )
    assert client.profile == loaded.profile == expected_profile
    assert client.no_sign_request == loaded.no_sign_request == expected_no_sign_request


def _assert_isinstance(obj, expected: type):
    assert isinstance(obj, expected), f"Expected a {expected}, got {type(obj)=}"
