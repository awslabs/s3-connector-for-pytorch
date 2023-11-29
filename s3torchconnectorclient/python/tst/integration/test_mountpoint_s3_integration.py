#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
import hashlib
import logging
import pickle
import math
import random
import sys
import uuid
import pytest

from s3torchconnectorclient._mountpoint_s3_client import (
    MountpointS3Client,
    S3Exception,
    ListObjectStream,
)

from conftest import get_test_config

logging.basicConfig(
    format="%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s"
)
logging.getLogger().setLevel(1)

log = logging.getLogger(__name__)

test_config = get_test_config()

HELLO_WORLD_DATA = b"Hello, World!\n"
TEST_USER_AGENT_PREFIX = "integration-tests"


@pytest.mark.parametrize(
    "region, bucket, key",
    [
        (test_config.region, test_config.bucket, "sample-files/hello_world.txt"),
        (
            test_config.express_region,
            test_config.express_bucket,
            "sample-files/hello_world.txt",
        ),
    ],
)
def test_get_object(region: str, bucket: str, key: str):
    client = MountpointS3Client(region, TEST_USER_AGENT_PREFIX)
    stream = client.get_object(bucket, key)

    full_data = b"".join(stream)
    assert full_data == HELLO_WORLD_DATA


@pytest.mark.parametrize(
    "region, bucket, key",
    [
        (test_config.region, test_config.bucket, "sample-files/hello_world.txt"),
        (
            test_config.express_region,
            test_config.express_bucket,
            "sample-files/hello_world.txt",
        ),
    ],
)
def test_get_object_with_unpickled_client(region: str, bucket: str, key: str):
    original_client = MountpointS3Client(region, TEST_USER_AGENT_PREFIX)
    assert original_client.user_agent_prefix == TEST_USER_AGENT_PREFIX
    pickled_client = pickle.dumps(original_client)
    assert isinstance(pickled_client, bytes)
    unpickled_client = pickle.loads(pickled_client)
    assert unpickled_client.user_agent_prefix == TEST_USER_AGENT_PREFIX
    stream = unpickled_client.get_object(
        bucket,
        key,
    )
    full_data = b"".join(stream)
    assert full_data == HELLO_WORLD_DATA


@pytest.mark.parametrize(
    "region, bucket, key",
    [
        (
            test_config.region,
            f"{test_config.bucket}-invalid",
            "sample-files/hello_world.txt",
        ),
        (
            test_config.express_region,
            f"{test_config.express_bucket}-invalid",
            "sample-files/hello_world.txt",
        ),
    ],
)
def test_get_object_invalid_bucket(region: str, bucket: str, key: str):
    client = MountpointS3Client(region, TEST_USER_AGENT_PREFIX)
    with pytest.raises(S3Exception) as error:
        next(client.get_object(f"{bucket}-{uuid.uuid4()}", key))
    assert str(error.value) == "Service error: The bucket does not exist"


@pytest.mark.parametrize(
    "region, bucket, key",
    [
        (test_config.region, test_config.bucket, "sample-files/not-hello_world.txt"),
        (
            test_config.express_region,
            test_config.express_bucket,
            "sample-files/not-hello_world.txt",
        ),
    ],
)
def test_get_object_invalid_key(region: str, bucket: str, key: str):
    client = MountpointS3Client(region, TEST_USER_AGENT_PREFIX)
    with pytest.raises(S3Exception) as error:
        next(client.get_object(bucket, key))
    assert str(error.value) == "Service error: The key does not exist"


@pytest.mark.parametrize(
    "region, bucket",
    [
        (test_config.region, test_config.bucket),
        (test_config.express_region, test_config.express_bucket),
    ],
)
def test_list_objects(region, bucket):
    client = MountpointS3Client(region, TEST_USER_AGENT_PREFIX)
    stream = client.list_objects(bucket)

    object_infos = [object_info for page in stream for object_info in page.object_info]
    keys = {object_info.key for object_info in object_infos}
    assert len(keys) > 1

    expected_img_10_keys = {f"e2e-tests/images-10/img{i}.jpg" for i in range(10)}
    assert keys > expected_img_10_keys


@pytest.mark.parametrize(
    "region, bucket, prefix",
    [
        (test_config.region, test_config.bucket, "e2e-tests/images-10/"),
        (
            test_config.express_region,
            test_config.express_bucket,
            "e2e-tests/images-10/",
        ),
    ],
)
def test_list_objects_with_prefix(region, bucket, prefix):
    client = MountpointS3Client(region, TEST_USER_AGENT_PREFIX)
    stream = client.list_objects(bucket, prefix)

    object_infos = [object_info for page in stream for object_info in page.object_info]
    keys = {object_info.key for object_info in object_infos}

    expected_img_10_keys = {f"e2e-tests/images-10/img{i}.jpg" for i in range(10)}
    assert keys == expected_img_10_keys


@pytest.mark.parametrize(
    "region, bucket, prefix",
    [
        (test_config.region, test_config.bucket, "e2e-tests/"),
        (test_config.express_region, test_config.express_bucket, "e2e-tests/"),
    ],
)
def test_multi_list_requests_return_same_list(region, bucket, prefix):
    client = MountpointS3Client(region, TEST_USER_AGENT_PREFIX)
    stream = client.list_objects(bucket, prefix)

    object_infos = [object_info for page in stream for object_info in page.object_info]
    keys_of_first_request = [object_info.key for object_info in object_infos]

    stream = client.list_objects(bucket, prefix)

    object_infos = [object_info for page in stream for object_info in page.object_info]
    keys_of_second_request = [object_info.key for object_info in object_infos]

    assert keys_of_first_request == keys_of_second_request


@pytest.mark.parametrize(
    "region, bucket, key, data_to_write",
    [
        (
            test_config.region,
            test_config.bucket,
            "put-integ-tests/single-part.txt",
            b"Hello, world!",
        ),
        (
            test_config.region,
            test_config.bucket,
            "put-integ-tests/empty-object.txt",
            b"!",
        ),
        (
            test_config.express_region,
            test_config.express_bucket,
            "put-integ-tests/single-part.txt",
            b"Hello, world!",
        ),
        (
            test_config.express_region,
            test_config.express_bucket,
            "put-integ-tests/empty-object.txt",
            b"!",
        ),
    ],
)
def test_put_object(region, bucket, key, data_to_write):
    client = MountpointS3Client(region, TEST_USER_AGENT_PREFIX)
    put_stream = client.put_object(bucket, key)

    put_stream.write(data_to_write)
    put_stream.close()

    get_stream = client.get_object(bucket, key)
    assert b"".join(get_stream) == data_to_write


@pytest.mark.parametrize(
    "region, bucket, key",
    [
        (test_config.region, test_config.bucket, "put-integ-tests/to-overwrite.txt"),
        (
            test_config.express_region,
            test_config.express_bucket,
            "put-integ-tests/to-overwrite.txt",
        ),
    ],
)
def test_put_object_overwrite(region: str, bucket: str, key: str):
    client = MountpointS3Client(region, TEST_USER_AGENT_PREFIX)

    put_stream = client.put_object(bucket, key)
    put_stream.write(b"before")
    put_stream.close()
    get_stream = client.get_object(bucket, key)
    assert b"".join(get_stream) == b"before"

    put_stream = client.put_object(bucket, key)

    put_stream.write(b"after")
    put_stream.close()

    get_stream = client.get_object(bucket, key)
    assert b"".join(get_stream) == b"after"


@pytest.mark.parametrize(
    "region, bucket, max_keys",
    [
        (test_config.region, test_config.bucket, 1),
        (test_config.region, test_config.bucket, 4),
        (test_config.region, test_config.bucket, 1000),
    ],
)
def test_s3_list_object_with_continuation(region: str, bucket: str, max_keys: int):
    client = MountpointS3Client(region, TEST_USER_AGENT_PREFIX)
    stream = client.list_objects(
        bucket, prefix="e2e-tests/images-10/", max_keys=max_keys
    )
    object_infos = _parse_list_result(stream, max_keys)
    keys = [object_info.key for object_info in object_infos]
    assert len(keys) > 1
    expected_img_10_keys = [f"e2e-tests/images-10/img{i}.jpg" for i in range(10)]
    assert keys == expected_img_10_keys


@pytest.mark.parametrize(
    "region, bucket, max_keys",
    [
        (test_config.express_region, test_config.express_bucket, 1),
        (test_config.express_region, test_config.express_bucket, 4),
        (test_config.express_region, test_config.express_bucket, 1000),
    ],
)
def test_s3express_list_object_with_continuation(
    region: str, bucket: str, max_keys: int
):
    client = MountpointS3Client(region, TEST_USER_AGENT_PREFIX)
    stream = client.list_objects(
        bucket, prefix="e2e-tests/images-10/", max_keys=max_keys
    )
    object_infos = _parse_list_result(stream, max_keys)
    initial_keys = [object_info.key for object_info in object_infos]
    assert len(initial_keys) > 1

    """
    For S3, list results are always returned in UTF-8 binary order.
    S3Express is returning list results in a consistent order, but not guaranteeing
    a sorted list as S3 does.
    """
    stream = client.list_objects(
        bucket, prefix="e2e-tests/images-10/", max_keys=max_keys
    )
    object_infos = _parse_list_result(stream, max_keys)
    keys = [object_info.key for object_info in object_infos]

    assert keys == initial_keys

    expected_img_10_keys = [f"e2e-tests/images-10/img{i}.jpg" for i in range(10)]
    assert sorted(keys) == expected_img_10_keys


@pytest.mark.parametrize(
    "region, bucket, key, part_count, part_size",
    [
        (
            test_config.region,
            test_config.bucket,
            "put-integ-tests/mpu-test.txt",
            1,
            5 * 1024 * 1024,
        ),
        (
            test_config.region,
            test_config.bucket,
            "put-integ-tests/mpu-test.txt",
            2,
            5 * 1024 * 1024,
        ),
        (
            test_config.express_region,
            test_config.express_bucket,
            "put-integ-tests/mpu-test.txt",
            1,
            5 * 1024 * 1024,
        ),
        (
            test_config.express_region,
            test_config.express_bucket,
            "put-integ-tests/mpu-test.txt",
            2,
            5 * 1024 * 1024,
        ),
    ],
)
def test_put_object_mpu(
    region: str, bucket: str, key: str, part_count: int, part_size: int
):
    data_to_write = randbytes(part_count * part_size)

    client = MountpointS3Client(region, TEST_USER_AGENT_PREFIX, part_size=part_size)

    put_stream = client.put_object(bucket, key)
    put_stream.write(data_to_write)
    put_stream.close()

    get_stream = client.get_object(bucket, key)
    assert b"".join(get_stream) == data_to_write


@pytest.mark.parametrize(
    "region, bucket, key, storage_class",
    [
        (test_config.region, test_config.bucket, "sample-files/hello_world.txt", None),
        (
            test_config.express_region,
            test_config.express_bucket,
            "sample-files/hello_world.txt",
            "EXPRESS_ONEZONE",
        ),
    ],
)
def test_head_object(region: str, bucket: str, key: str, storage_class: str):
    client = MountpointS3Client(region, TEST_USER_AGENT_PREFIX)
    object_info = client.head_object(
        bucket,
        key,
    )

    assert object_info.size == len(HELLO_WORLD_DATA)
    assert object_info.restore_status is None
    assert object_info.storage_class == storage_class
    assert object_info.etag is not None
    if bucket == test_config.bucket:
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
