import logging

import pytest
from s3dataset._s3dataset import MockMountpointS3Client

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
    client = MockMountpointS3Client(REGION, MOCK_BUCKET, part_size)
    client.add_object(key, data)
    stream = client.get_object(MOCK_BUCKET, key)

    returned_data = b''.join(stream)
    assert returned_data == data


@pytest.mark.parametrize(
    "expected_keys",
    [
        ({"test"}),
        ({"multiple", "objects"}),
        (set()),
    ],
)
def test_list_objects(expected_keys):
    client = MockMountpointS3Client(REGION, MOCK_BUCKET)
    for key in expected_keys:
        client.add_object(key, b"")

    stream = client.list_objects(MOCK_BUCKET)

    object_infos = [object_info for page in stream for object_info in page.object_info]
    keys = {object_info.key for object_info in object_infos}
    assert keys == expected_keys
