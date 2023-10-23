import logging

import pytest

from unittest.mock import Mock
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
    "object_info, stream",
    [
        (None, None),
        (MOCK_OBJECT_INFO, None),
        (None, MOCK_STREAM),
        (TEST_BUCKET, None),
        (TEST_BUCKET, ""),
    ],
)
def test_s3object_creation(object_info, stream):
    s3object = S3Object(TEST_BUCKET, TEST_KEY, object_info, stream)
    assert s3object
    assert s3object.bucket == TEST_BUCKET
    assert s3object.key == TEST_KEY
    assert s3object.object_info == object_info
    assert s3object.stream == stream


@pytest.mark.parametrize(
    "bucket, key",
    [(None, None), (None, ""), (None, TEST_KEY), ("", TEST_KEY)],
)
def test_s3object_invalid_creation(bucket, key):
    with pytest.raises(ValueError) as error:
        S3Object(bucket, key)
    assert str(error.value) == "Bucket should be specified"
