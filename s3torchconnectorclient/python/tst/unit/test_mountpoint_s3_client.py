#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import logging
import pickle
from collections import namedtuple

import pytest
from typing import Set, Optional

from s3torchconnectorclient import __version__
from s3torchconnectorclient._mountpoint_s3_client import (
    S3Exception,
    GetObjectStream,
    ListObjectStream,
    PutObjectStream,
    MockMountpointS3Client,
    MountpointS3Client,
)

logging.basicConfig(
    format="%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s"
)

REGION = "us-east-1"
MOCK_BUCKET = "mock-bucket"
INVALID_ENDPOINT = "INVALID"


@pytest.mark.parametrize(
    "key, data, part_size",
    [
        ("hello_world.txt", b"Hello, world!", 1000),
        ("multipart", b"The quick brown fox jumps over the lazy dog.", 2),
    ],
)
@pytest.mark.parametrize("force_path_style", [False, True])
def test_get_object(key: str, data: bytes, part_size: int, force_path_style: bool):
    mock_client = MockMountpointS3Client(
        REGION, MOCK_BUCKET, part_size=part_size, force_path_style=force_path_style
    )
    mock_client.add_object(key, data)
    client = mock_client.create_mocked_client()

    stream = client.get_object(MOCK_BUCKET, key)
    _assert_isinstance(stream, GetObjectStream)

    returned_data = b"".join(stream)
    assert returned_data == data


@pytest.mark.parametrize("force_path_style", [False, True])
def test_get_object_part_size(force_path_style: bool):
    mock_client = MockMountpointS3Client(
        REGION, MOCK_BUCKET, part_size=2, force_path_style=force_path_style
    )
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


@pytest.mark.parametrize("force_path_style", [False, True])
def test_get_object_bad_bucket(force_path_style: bool):
    mock_client = MockMountpointS3Client(
        REGION, MOCK_BUCKET, force_path_style=force_path_style
    )
    client = mock_client.create_mocked_client()

    with pytest.raises(S3Exception, match="Service error: The bucket does not exist"):
        client.get_object("does_not_exist", "foo")


@pytest.mark.parametrize("force_path_style", [False, True])
def test_get_object_none_bucket(force_path_style: bool):
    mock_client = MockMountpointS3Client(
        REGION, MOCK_BUCKET, force_path_style=force_path_style
    )
    client = mock_client.create_mocked_client()

    with pytest.raises(TypeError):
        client.get_object(None, "foo")


@pytest.mark.parametrize("force_path_style", [False, True])
def test_get_object_bad_object(force_path_style: bool):
    mock_client = MockMountpointS3Client(
        REGION, MOCK_BUCKET, force_path_style=force_path_style
    )
    client = mock_client.create_mocked_client()

    with pytest.raises(S3Exception, match="Service error: The key does not exist"):
        client.get_object(MOCK_BUCKET, "does_not_exist")


@pytest.mark.parametrize("force_path_style", [False, True])
def test_get_object_iterates_once(force_path_style: bool):
    mock_client = MockMountpointS3Client(
        REGION, MOCK_BUCKET, force_path_style=force_path_style
    )
    mock_client.add_object("key", b"data")
    client = mock_client.create_mocked_client()

    stream = client.get_object(MOCK_BUCKET, "key")
    _assert_isinstance(stream, GetObjectStream)

    returned_data = b"".join(stream)
    assert returned_data == b"data"

    returned_data = b"".join(stream)
    assert returned_data == b""


@pytest.mark.parametrize("force_path_style", [False, True])
def test_get_object_throws_stop_iteration(force_path_style: bool):
    mock_client = MockMountpointS3Client(
        REGION, MOCK_BUCKET, force_path_style=force_path_style
    )
    mock_client.add_object("key", b"data")
    client = mock_client.create_mocked_client()

    stream = client.get_object(MOCK_BUCKET, "key")
    _assert_isinstance(stream, GetObjectStream)

    for _ in stream:
        pass

    for _ in range(10):
        with pytest.raises(StopIteration):
            next(stream)


@pytest.mark.parametrize(
    "expected_keys",
    [
        ({"tests"}),
        ({"multiple", "objects"}),
        (set()),
    ],
)
@pytest.mark.parametrize("force_path_style", [False, True])
def test_list_objects(expected_keys: Set[str], force_path_style: bool):
    mock_client = MockMountpointS3Client(
        REGION, MOCK_BUCKET, force_path_style=force_path_style
    )
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
@pytest.mark.parametrize("force_path_style", [False, True])
def test_list_objects_with_prefix(
    prefix: str, keys: Set[str], expected_keys: Set[str], force_path_style: bool
):
    mock_client = MockMountpointS3Client(
        REGION, MOCK_BUCKET, force_path_style=force_path_style
    )
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
@pytest.mark.parametrize("force_path_style", [False, True])
def test_put_object(data_to_write: bytes, part_size: int, force_path_style: bool):
    mock_client = MockMountpointS3Client(
        REGION, MOCK_BUCKET, part_size=part_size, force_path_style=force_path_style
    )
    client = mock_client.create_mocked_client()

    put_stream = client.put_object(MOCK_BUCKET, "key")
    assert isinstance(put_stream, PutObjectStream)

    put_stream.write(data_to_write)
    put_stream.close()

    get_stream = client.get_object(MOCK_BUCKET, "key")
    assert b"".join(get_stream) == data_to_write


@pytest.mark.parametrize("force_path_style", [False, True])
def test_put_object_overwrite(force_path_style: bool):
    mock_client = MockMountpointS3Client(
        REGION, MOCK_BUCKET, force_path_style=force_path_style
    )
    mock_client.add_object("key", b"before")
    client = mock_client.create_mocked_client()

    put_stream = client.put_object(MOCK_BUCKET, "key")
    assert isinstance(put_stream, PutObjectStream)

    put_stream.write(b"after")
    put_stream.close()

    get_stream = client.get_object(MOCK_BUCKET, "key")
    assert b"".join(get_stream) == b"after"


@pytest.mark.parametrize("force_path_style", [False, True])
def test_put_object_no_multiple_close(force_path_style: bool):
    mock_client = MockMountpointS3Client(
        REGION, MOCK_BUCKET, force_path_style=force_path_style
    )
    client = mock_client.create_mocked_client()

    put_stream = client.put_object(MOCK_BUCKET, "key")
    assert isinstance(put_stream, PutObjectStream)

    put_stream.write(b"")
    put_stream.close()
    with pytest.raises(S3Exception, match="Cannot close object more than once"):
        put_stream.close()


@pytest.mark.parametrize("force_path_style", [False, True])
def test_put_object_no_write_after_close(force_path_style: bool):
    mock_client = MockMountpointS3Client(
        REGION, MOCK_BUCKET, force_path_style=force_path_style
    )
    client = mock_client.create_mocked_client()

    put_stream = client.put_object(MOCK_BUCKET, "key")
    assert isinstance(put_stream, PutObjectStream)

    put_stream.write(b"")
    put_stream.close()
    with pytest.raises(S3Exception, match="Cannot write to closed object"):
        put_stream.write(b"")


@pytest.mark.parametrize("force_path_style", [False, True])
def test_put_object_with_storage_class(force_path_style: bool):
    mock_client = MockMountpointS3Client(
        REGION, MOCK_BUCKET, force_path_style=force_path_style
    )
    client = mock_client.create_mocked_client()

    put_stream = client.put_object(MOCK_BUCKET, "key", "STANDARD_IA")
    assert isinstance(put_stream, PutObjectStream)

    put_stream.write(b"")
    put_stream.close()


# TODO: Add hypothesis setup after aligning on limits
def test_mountpoint_client_pickles():
    expected_profile = None
    expected_unsigned = False
    expected_region = REGION
    expected_part_size = 5 * 2**20
    expected_throughput_target_gbps = 3.5
    expected_force_path_style = True

    client = MountpointS3Client(
        region=expected_region,
        user_agent_prefix="unit-tests",
        part_size=expected_part_size,
        throughput_target_gbps=expected_throughput_target_gbps,
        profile=expected_profile,
        unsigned=expected_unsigned,
        force_path_style=expected_force_path_style,
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
    assert client.unsigned == loaded.unsigned == expected_unsigned
    assert (
        client.force_path_style == loaded.force_path_style == expected_force_path_style
    )


@pytest.mark.parametrize(
    "endpoint, expected",
    [
        ("https://s3.us-east-1.amazonaws.com", "https://s3.us-east-1.amazonaws.com"),
        ("", ""),
        (None, None),
        ("https://us-east-1.amazonaws.com", "https://us-east-1.amazonaws.com"),
    ],
)
@pytest.mark.parametrize("force_path_style", [False, True])
def test_mountpoint_client_creation_with_region_and_endpoint(
    endpoint: Optional[str],
    expected: Optional[str],
    force_path_style: bool,
):
    client = MountpointS3Client(
        region=REGION, endpoint=endpoint, force_path_style=force_path_style
    )
    assert isinstance(client, MountpointS3Client)
    assert client.endpoint == expected
    assert client.force_path_style == force_path_style


def test_mountpoint_client_creation_with_region_and_invalid_endpoint():
    client = MountpointS3Client(region=REGION, endpoint=INVALID_ENDPOINT)
    assert isinstance(client, MountpointS3Client)
    assert client.endpoint == INVALID_ENDPOINT
    with pytest.raises(S3Exception) as e:
        put_stream = client.put_object(MOCK_BUCKET, "key")
    assert (
        str(e.value)
        == "Client error: Failed to construct request: Invalid S3 endpoint: endpoint could not be resolved: Custom endpoint `INVALID` was not a valid URI"
    )


@pytest.mark.parametrize("force_path_style", [False, True])
def test_delete_object(force_path_style: bool):
    key = "hello_world.txt"
    mock_client = MockMountpointS3Client(
        REGION, MOCK_BUCKET, force_path_style=force_path_style
    )
    mock_client.add_object(key, b"data")
    client = mock_client.create_mocked_client()

    client.delete_object(MOCK_BUCKET, key)

    with pytest.raises(S3Exception, match="Service error: The key does not exist"):
        client.get_object(MOCK_BUCKET, key)


@pytest.mark.parametrize("force_path_style", [False, True])
def test_delete_object_already_deleted(force_path_style: bool):
    mock_client = MockMountpointS3Client(
        REGION, MOCK_BUCKET, force_path_style=force_path_style
    )
    client = mock_client.create_mocked_client()

    client.delete_object(MOCK_BUCKET, "hello_world.txt")


@pytest.mark.parametrize("force_path_style", [False, True])
def test_delete_object_non_existent_bucket(force_path_style: bool):
    mock_client = MockMountpointS3Client(REGION, MOCK_BUCKET, force_path_style=force_path_style)
    client = mock_client.create_mocked_client()

    with pytest.raises(S3Exception, match="Service error: The bucket does not exist"):
        client.delete_object("bucket2", "hello_world.txt")


# NOTE 1: `MockMountpointS3Client` only works on one bucket, so it cannot test
# cross-bucket COPY operations.
# NOTE 2: as of Oct'24, `force_path_style` is an unsupported option
# (https://github.com/awslabs/aws-c-s3/blob/main/include/aws/s3/s3_client.h#L74-L86).
def test_copy_object():
    S3Object = namedtuple('S3Object', ['key', 'data'])
    src = S3Object("src_key.txt", b"src_data")
    dst = S3Object("dst_key.txt", None)

    mock_client = MockMountpointS3Client(REGION, MOCK_BUCKET)
    mock_client.add_object(src.key, src.data)

    client = mock_client.create_mocked_client()
    client.copy_object(MOCK_BUCKET, src.key, MOCK_BUCKET, dst.key)

    dst_stream = client.get_object(MOCK_BUCKET, dst.key)
    assert dst_stream.key == dst.key
    assert b''.join(dst_stream) == src.data


# NOTE 1: `MockMountpointS3Client` only works on one bucket, so it cannot test
# cross-bucket COPY operations.
# NOTE 2: as of Oct'24, `force_path_style` is an unsupported option
# (https://github.com/awslabs/aws-c-s3/blob/main/include/aws/s3/s3_client.h#L74-L86).
def test_copy_object_to_non_existent_bucket():
    S3Object = namedtuple('S3Object', ['key', 'data'])
    src = S3Object("src_key.txt", b"src_data")
    dst = S3Object("dst_key.txt", None)

    mock_client = MockMountpointS3Client(REGION, MOCK_BUCKET)
    mock_client.add_object(src.key, src.data)

    client = mock_client.create_mocked_client()

    with pytest.raises(S3Exception, match="Service error: The bucket does not exist"):
        client.copy_object(MOCK_BUCKET, src.key, "bucket2", dst.key)


def _assert_isinstance(obj, expected: type):
    assert isinstance(obj, expected), f"Expected a {expected}, got {type(obj)=}"


def test_client_version():
    assert isinstance(__version__, str)
    assert __version__ > "1.0.0"
