from io import BytesIO

import torch
from s3dataset_s3_client._s3dataset import MockMountpointS3Client

from s3dataset_s3_client import S3Object
from s3dataset_s3_client.put_object_stream_wrapper import PutObjectStreamWrapper

TEST_BUCKET = "test-bucket"
TEST_KEY = "test-key"
TEST_REGION = "us-east-1"


def test_general_checkpointing_saves():
    mock_client = MockMountpointS3Client(TEST_REGION, TEST_BUCKET)
    client = mock_client.create_mocked_client()

    with PutObjectStreamWrapper(
        client.put_object(TEST_BUCKET, TEST_KEY)
    ) as put_object_request:
        torch.save({}, put_object_request)

    serialised = BytesIO(b"".join(client.get_object(TEST_BUCKET, TEST_KEY)))
    assert torch.load(serialised) == {}


def test_general_checkpointing_loads():
    mock_client = MockMountpointS3Client(TEST_REGION, TEST_BUCKET)
    serialised = BytesIO()
    torch.save({}, serialised)
    serialised_size = serialised.tell()
    serialised.seek(0)
    mock_client.add_object(TEST_KEY, serialised.read())

    client = mock_client.create_mocked_client()
    s3object = S3Object(
        TEST_BUCKET,
        TEST_KEY,
        get_stream=lambda: client.get_object(TEST_BUCKET, TEST_KEY),
    )
    # TODO - mock HeadObject to do the size fetching properly.
    s3object._size = serialised_size

    assert torch.load(s3object) == {}
