import logging
from unittest.mock import Mock

import pytest

from s3dataset._s3dataset import ObjectInfo, GetObjectStream
from s3dataset.s3object import S3Object

logging.basicConfig(
    format="%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s"
)
logging.getLogger().setLevel(1)

log = logging.getLogger(__name__)


TEST_BUCKET = "test-bucket"
TEST_KEY = "test-key"
MOCK_OBJECT_INFO = Mock(ObjectInfo)
MOCK_STREAM = Mock(GetObjectStream)


@pytest.mark.parametrize(
    "object_info, get_stream",
    [
        (None, lambda: None),
        (MOCK_OBJECT_INFO, lambda: None),
        (None, lambda: MOCK_STREAM),
        (TEST_BUCKET, lambda: None),
        (TEST_BUCKET, lambda: ""),
    ],
)
def test_s3object_creation(object_info, get_stream):
    s3object = S3Object(TEST_BUCKET, TEST_KEY, object_info, get_stream)
    assert s3object
    assert s3object.bucket == TEST_BUCKET
    assert s3object.key == TEST_KEY
    assert s3object.object_info == object_info
    assert s3object._get_stream is get_stream


@pytest.mark.parametrize(
    "bucket, key",
    [(None, None), (None, ""), (None, TEST_KEY), ("", TEST_KEY)],
)
def test_s3object_invalid_creation(bucket, key):
    with pytest.raises(ValueError) as error:
        S3Object(bucket, key)
    assert str(error.value) == "Bucket should be specified"


@pytest.mark.parametrize(
    "stream",
    [
        [b"1", b"2", b"3"],
        [],
        [b"hello!"],
    ],
)
def test_s3object_prefetch(stream):
    s3object = S3Object(TEST_BUCKET, TEST_KEY, None, lambda: stream)
    assert s3object._stream is None
    s3object.prefetch()
    assert list(s3object._stream) == stream
    s3object.prefetch()
    assert list(s3object._stream) == []


@pytest.mark.parametrize(
    "stream",
    [
        [b"1", b"2", b"3"],
        [],
        [b"hello!"],
    ],
)
def test_s3object_read(stream):
    data_received_cb = []
    s3object = S3Object(
        TEST_BUCKET,
        TEST_KEY,
        None,
        lambda: stream,
        lambda s3object, data: data_received_cb.append((s3object, data)),
    )
    assert s3object._stream is None
    assert b"".join(stream) == s3object.read()
    assert [data for _, data in data_received_cb] == stream
    assert all(obj is s3object for obj, _ in data_received_cb)
