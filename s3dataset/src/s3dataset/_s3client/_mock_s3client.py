from . import S3Client
from s3dataset_s3_client._s3dataset import MockMountpointS3Client


class MockS3Client(S3Client):
    def __init__(self, region: str, bucket: str, part_size: int = 8 * 1024 * 1024):
        self._mock_client = MockMountpointS3Client(region, bucket, part_size=part_size)
        self._client = self._mock_client.create_mocked_client()

    def add_object(self, key: str, data: bytes) -> None:
        self._mock_client.add_object(key, data)

    def remove_object(self, key: str) -> None:
        self._mock_client.remove_object(key)
