import logging

import pytest
from s3dataset._s3dataset import S3DatasetException, GetObjectStream, ListObjectStream, MountpointS3Client, \
    MountpointS3ClientMock

logging.basicConfig(format='%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s')
logging.getLogger().setLevel(1)

log = logging.getLogger(__name__)


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
    mock_client = MountpointS3ClientMock(REGION, MOCK_BUCKET, part_size=part_size)
    client = MountpointS3Client.with_client(mock_client)

    mock_client.add_object(key, data)

    stream = client.get_object(MOCK_BUCKET, key)
    _assert_isinstance(stream, GetObjectStream)

    returned_data = b''.join(stream)
    assert returned_data == data


def test_get_object_part_size():
    mock_client = MountpointS3ClientMock(REGION, MOCK_BUCKET, part_size=2)
    client = MountpointS3Client.with_client(mock_client)

    mock_client.add_object("key", b"1234567890")

    stream = client.get_object(MOCK_BUCKET, "key")
    _assert_isinstance(stream, GetObjectStream)

    returned_data = list(stream)
    assert returned_data == [b"12", b"34", b"56", b"78", b"90"]


def test_get_object_bad_bucket():
    mock_client = MountpointS3ClientMock(REGION, MOCK_BUCKET)
    client = MountpointS3Client.with_client(mock_client)

    try:
        client.get_object("does_not_exist", "foo")
    except S3DatasetException as e:
        assert str(e) == "Service error: The bucket does not exist"


def test_get_object_none_bucket():
    mock_client = MountpointS3ClientMock(REGION, MOCK_BUCKET)
    client = MountpointS3Client.with_client(mock_client)

    try:
        client.get_object(None, "foo")
    except TypeError:
        pass
    else:
        raise AssertionError("Should raise TypeError")


def test_get_object_bad_object():
    mock_client = MountpointS3ClientMock(REGION, MOCK_BUCKET)
    client = MountpointS3Client.with_client(mock_client)

    try:
        client.get_object(MOCK_BUCKET, "does_not_exist")
    except S3DatasetException as e:
        assert str(e) == "Service error: The key does not exist"


def test_get_object_iterates_once():
    mock_client = MountpointS3ClientMock(REGION, MOCK_BUCKET)
    client = MountpointS3Client.with_client(mock_client)

    mock_client.add_object("key", b"data")

    stream = client.get_object(MOCK_BUCKET, "key")
    _assert_isinstance(stream, GetObjectStream)

    returned_data = b''.join(stream)
    assert returned_data == b"data"

    returned_data = b''.join(stream)
    assert returned_data == b""


def test_get_object_throws_stop_iteration():
    mock_client = MountpointS3ClientMock(REGION, MOCK_BUCKET)
    client = MountpointS3Client.with_client(mock_client)

    mock_client.add_object("key", b"data")

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
    mock_client = MountpointS3ClientMock(REGION, MOCK_BUCKET)
    client = MountpointS3Client.with_client(mock_client)

    for key in expected_keys:
        mock_client.add_object(key, b"")

    stream = client.list_objects(MOCK_BUCKET)
    assert isinstance(stream, ListObjectStream)

    object_infos = [object_info for page in stream for object_info in page.object_info]
    keys = {object_info.key for object_info in object_infos}
    assert keys == expected_keys


def _assert_isinstance(obj, expected: type):
    assert isinstance(obj, expected), f"Expected a {expected}, got {type(obj)=}"
