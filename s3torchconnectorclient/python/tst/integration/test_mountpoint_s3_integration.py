#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
import hashlib
import logging
import math
import pickle
import sys
import uuid
import random
import pytest

from s3torchconnectorclient._mountpoint_s3_client import (
    MountpointS3Client,
    S3Exception,
    ListObjectStream,
)

logging.basicConfig(
    format="%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s"
)
logging.getLogger().setLevel(1)

log = logging.getLogger(__name__)

HELLO_WORLD_DATA = b"Hello, World!\n"
TEST_USER_AGENT_PREFIX = "integration-tests"


def test_get_object(sample_directory):
    client = MountpointS3Client(sample_directory.region, TEST_USER_AGENT_PREFIX)
    stream = client.get_object(
        sample_directory.bucket, f"{sample_directory.prefix}hello_world.txt"
    )

    full_data = b"".join(stream)
    assert full_data == HELLO_WORLD_DATA


def test_get_object_with_endpoint(sample_directory):
    client = MountpointS3Client(
        sample_directory.region, TEST_USER_AGENT_PREFIX, endpoint=CI_CUSTOM_ENDPOINT_URL
    )
    stream = client.get_object(
        sample_directory.bucket, f"{sample_directory.prefix}hello_world.txt"
    )

    full_data = b"".join(stream)
    assert full_data == HELLO_WORLD_DATA


def test_get_object_with_unpickled_client(sample_directory):
    original_client = MountpointS3Client(
        sample_directory.region, TEST_USER_AGENT_PREFIX
    )
    assert original_client.user_agent_prefix == TEST_USER_AGENT_PREFIX
    pickled_client = pickle.dumps(original_client)
    assert isinstance(pickled_client, bytes)
    unpickled_client = pickle.loads(pickled_client)
    assert unpickled_client.user_agent_prefix == TEST_USER_AGENT_PREFIX
    stream = unpickled_client.get_object(
        sample_directory.bucket, f"{sample_directory.prefix}hello_world.txt"
    )
    full_data = b"".join(stream)
    assert full_data == HELLO_WORLD_DATA


def test_get_object_invalid_bucket(sample_directory):
    client = MountpointS3Client(sample_directory.region, TEST_USER_AGENT_PREFIX)
    with pytest.raises(S3Exception) as error:
        next(
            client.get_object(
                f"{sample_directory.bucket}-{uuid.uuid4()}", sample_directory.prefix
            )
        )
    assert str(error.value) == "Service error: The bucket does not exist"


def test_get_object_invalid_prefix(sample_directory):
    client = MountpointS3Client(sample_directory.region, TEST_USER_AGENT_PREFIX)
    with pytest.raises(S3Exception) as error:
        next(
            client.get_object(
                sample_directory.bucket, f"{sample_directory.prefix}-{uuid.uuid4()}"
            )
        )
    assert str(error.value) == "Service error: The key does not exist"


def test_list_objects(image_directory):
    client = MountpointS3Client(image_directory.region, TEST_USER_AGENT_PREFIX)
    stream = client.list_objects(image_directory.bucket)

    object_infos = [object_info for page in stream for object_info in page.object_info]
    keys = {object_info.key for object_info in object_infos}
    assert len(keys) > 1

    expected_img_10_keys = {
        f"{image_directory.prefix}img{i:03d}.jpg" for i in range(10)
    }
    assert keys > expected_img_10_keys


def test_list_objects_with_prefix(image_directory):
    client = MountpointS3Client(image_directory.region, TEST_USER_AGENT_PREFIX)
    stream = client.list_objects(image_directory.bucket, image_directory.prefix)

    object_infos = [object_info for page in stream for object_info in page.object_info]
    keys = {object_info.key for object_info in object_infos}

    expected_img_10_keys = {
        f"{image_directory.prefix}img{i:03d}.jpg" for i in range(10)
    }
    assert keys == expected_img_10_keys


def test_multi_list_requests_return_same_list(image_directory):
    client = MountpointS3Client(image_directory.region, TEST_USER_AGENT_PREFIX)
    stream = client.list_objects(image_directory.bucket, image_directory.prefix)

    object_infos = [object_info for page in stream for object_info in page.object_info]
    keys_of_first_request = [object_info.key for object_info in object_infos]

    stream = client.list_objects(image_directory.bucket, image_directory.prefix)

    object_infos = [object_info for page in stream for object_info in page.object_info]
    keys_of_second_request = [object_info.key for object_info in object_infos]

    assert keys_of_first_request == keys_of_second_request


@pytest.mark.parametrize(
    "filename, content",
    [
        (
            "single-part.txt",
            b"Hello, world!",
        ),
        (
            "empty-object.txt",
            b"!",
        ),
    ],
)
def test_put_object(filename: str, content: bytes, put_object_tests_directory):
    client = MountpointS3Client(
        put_object_tests_directory.region, TEST_USER_AGENT_PREFIX
    )
    put_stream = client.put_object(
        put_object_tests_directory.bucket,
        f"{put_object_tests_directory.prefix}{filename}",
    )

    put_stream.write(content)
    put_stream.close()

    get_stream = client.get_object(
        put_object_tests_directory.bucket,
        f"{put_object_tests_directory.prefix}{filename}",
    )
    assert b"".join(get_stream) == content


def test_put_object_overwrite(put_object_tests_directory):
    client = MountpointS3Client(
        put_object_tests_directory.region, TEST_USER_AGENT_PREFIX
    )

    put_stream = client.put_object(
        put_object_tests_directory.bucket, put_object_tests_directory.prefix
    )
    put_stream.write(b"before")
    put_stream.close()
    get_stream = client.get_object(
        put_object_tests_directory.bucket, put_object_tests_directory.prefix
    )
    assert b"".join(get_stream) == b"before"

    put_stream = client.put_object(
        put_object_tests_directory.bucket, put_object_tests_directory.prefix
    )

    put_stream.write(b"after")
    put_stream.close()

    get_stream = client.get_object(
        put_object_tests_directory.bucket, put_object_tests_directory.prefix
    )
    assert b"".join(get_stream) == b"after"


@pytest.mark.parametrize("max_keys", {1, 4, 1000})
def test_s3_list_object_with_continuation(max_keys: int, image_directory):
    client = MountpointS3Client(image_directory.region, TEST_USER_AGENT_PREFIX)
    stream = client.list_objects(
        image_directory.bucket, prefix=image_directory.prefix, max_keys=max_keys
    )
    object_infos = _parse_list_result(stream, max_keys)
    initial_keys = [object_info.key for object_info in object_infos]
    assert len(initial_keys) > 1

    expected_img_10_keys = [
        f"{image_directory.prefix}img{i:03d}.jpg" for i in range(10)
    ]

    if image_directory.storage_class == "EXPRESS_ONEZONE":
        # S3Express is returning list results in a consistent order, but does not guarantee a sorted list as S3 does.
        stream = client.list_objects(
            image_directory.bucket, prefix=image_directory.prefix, max_keys=max_keys
        )
        object_infos = _parse_list_result(stream, max_keys)
        keys = [object_info.key for object_info in object_infos]
        assert keys == initial_keys
        assert sorted(keys) == expected_img_10_keys
    else:
        #  For S3, list results are always returned in UTF-8 binary order.
        assert initial_keys == expected_img_10_keys


@pytest.mark.parametrize(
    "part_count, part_size",
    [
        (
            1,
            5 * 1024 * 1024,
        ),
        (
            2,
            5 * 1024 * 1024,
        ),
    ],
)
def test_put_object_mpu(part_count: int, part_size: int, put_object_tests_directory):
    data_to_write = randbytes(part_count * part_size)

    client = MountpointS3Client(
        put_object_tests_directory.region, TEST_USER_AGENT_PREFIX, part_size=part_size
    )

    put_stream = client.put_object(
        put_object_tests_directory.bucket,
        f"{put_object_tests_directory.prefix}mpu-test.txt",
    )
    put_stream.write(data_to_write)
    put_stream.close()

    get_stream = client.get_object(
        put_object_tests_directory.bucket,
        f"{put_object_tests_directory.prefix}mpu-test.txt",
    )
    assert b"".join(get_stream) == data_to_write


def test_head_object(sample_directory):
    client = MountpointS3Client(sample_directory.region, TEST_USER_AGENT_PREFIX)
    object_info = client.head_object(
        sample_directory.bucket,
        f"{sample_directory.prefix}hello_world.txt",
    )

    assert object_info.size == len(HELLO_WORLD_DATA)
    assert object_info.restore_status is None
    assert object_info.etag is not None
    if sample_directory.storage_class == "EXPRESS_ONEZONE":
        # S3 Express has storage class EXPRESS_ONEZONE
        assert object_info.storage_class == sample_directory.storage_class
    else:
        assert object_info.storage_class is None
        object_md5 = hashlib.md5(HELLO_WORLD_DATA).hexdigest()
        expected_etag = f'"{object_md5}"'
        assert object_info.etag == expected_etag


def _parse_list_result(stream: ListObjectStream, max_keys: int):
    object_infos = []
    i = 0
    for i, page in enumerate(stream):
        for object_info in page.object_info:
            object_infos.append(object_info)
    assert (i + 1) == math.ceil(len(object_infos) / max_keys)
    return object_infos


def randbytes(n):
    return random.getrandbits(n * 8).to_bytes(n, sys.byteorder)
