import logging

import pytest

from s3dataset._s3dataset import (
    S3DatasetException,
    GetObjectStream,
    ListObjectStream,
    PutObjectStream,
    MockMountpointS3Client,
)

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
    mock_client = MockMountpointS3Client(REGION, MOCK_BUCKET, part_size=part_size)
    mock_client.add_object(key, data)
    client = mock_client.create_mocked_client()

    stream = client.get_object(MOCK_BUCKET, key)
    _assert_isinstance(stream, GetObjectStream)

    returned_data = b''.join(stream)
    assert returned_data == data


def test_get_object_part_size():
    mock_client = MockMountpointS3Client(REGION, MOCK_BUCKET, part_size=2)
    mock_client.add_object("key", b"1234567890")
    client = mock_client.create_mocked_client()

    stream = client.get_object(MOCK_BUCKET, "key")
    _assert_isinstance(stream, GetObjectStream)

    returned_data = list(stream)
    assert returned_data == [b"12", b"34", b"56", b"78", b"90"]


def test_get_object_bad_bucket():
    mock_client = MockMountpointS3Client(REGION, MOCK_BUCKET)
    client = mock_client.create_mocked_client()

    try:
        client.get_object("does_not_exist", "foo")
    except S3DatasetException as e:
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
    except S3DatasetException as e:
        assert str(e) == "Service error: The key does not exist"


def test_get_object_iterates_once():
    mock_client = MockMountpointS3Client(REGION, MOCK_BUCKET)
    mock_client.add_object("key", b"data")
    client = mock_client.create_mocked_client()

    stream = client.get_object(MOCK_BUCKET, "key")
    _assert_isinstance(stream, GetObjectStream)

    returned_data = b''.join(stream)
    assert returned_data == b"data"

    returned_data = b''.join(stream)
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
    "data_to_write,part_size",
    [
        (b"Hello, world!", 2000),
        (b"MultiPartUpload", 2)
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
    assert b''.join(get_stream) == data_to_write


def test_put_object_no_multiple_close():
    mock_client = MockMountpointS3Client(REGION, MOCK_BUCKET)
    client = mock_client.create_mocked_client()

    put_stream = client.put_object(MOCK_BUCKET, "key")
    assert isinstance(put_stream, PutObjectStream)

    put_stream.write(b"")
    put_stream.close()
    try:
        put_stream.close()
    except S3DatasetException as e:
        assert str(e) == "Cannot close object more than once"


def test_put_object_no_write_after_close():
    mock_client = MockMountpointS3Client(REGION, MOCK_BUCKET)
    client = mock_client.create_mocked_client()

    put_stream = client.put_object(MOCK_BUCKET, "key")
    assert isinstance(put_stream, PutObjectStream)

    put_stream.write(b"")
    put_stream.close()
    try:
        put_stream.write(b"")
    except S3DatasetException as e:
        assert str(e) == "Cannot write to closed object"


def test_put_object_with_storage_class():
    mock_client = MockMountpointS3Client(REGION, MOCK_BUCKET)
    client = mock_client.create_mocked_client()

    put_stream = client.put_object(MOCK_BUCKET, "key", "STANDARD_IA")
    assert isinstance(put_stream, PutObjectStream)

    put_stream.write(b"")
    put_stream.close()


def _assert_isinstance(obj, expected: type):
    assert isinstance(obj, expected), f"Expected a {expected}, got {type(obj)=}"
