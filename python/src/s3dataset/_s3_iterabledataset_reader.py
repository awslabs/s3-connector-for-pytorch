import io
from typing import Iterable

import torch
from s3dataset._s3dataset import MountpointS3Client, ObjectInfo, GetObjectStream


class S3DatasetObject(io.BufferedIOBase):
    def __init__(self, bucket: str, key: str, object_info: ObjectInfo=None, stream: GetObjectStream=None):
        self.bucket = bucket
        self.key = key
        self.object_info = object_info
        self.stream = stream

    # TODO: How to check we need to stop? What other checks do we need to add?
    def read(self, size=-1):
        return b''.join(self.stream)
        pass

    # TODO: Same here
    def readline(self, **kwargs):
        result = self.tell()
        self.__next__()
        return result
        pass

class S3DatasetSource():
    def __init__(self):
        self.dataset_objects = []

    def from_object_uris(self, object_uris: Iterable[str]):
        self.dataset_objects = map(self._uri_to_dataset_object, object_uris)
        return self

    # TODO: Update with prefix support
    def from_bucket(self, client: MountpointS3Client, bucket: str):
        self.dataset_objects = self._list_objects_for_bucket(client, bucket)
        return self

    def _uri_to_dataset_object(self, object_uri: str):
        bucket, key = self._parse_s3_uri(object_uri)
        return S3DatasetObject(bucket, key)

    def _list_objects_for_bucket(self, client: MountpointS3Client, bucket: str) -> Iterable[S3DatasetObject]:
        stream = client.list_objects(bucket)
        return [S3DatasetObject(bucket, object_info.key, object_info) for page in stream for object_info in page.object_info]

    def _parse_s3_uri(self, uri: str) -> [str, str]:
        # TODO: We should be able to support more through Mountpoint, not sure if we want to
        if not uri.startswith("s3://"):
            raise ValueError("Only s3:// URIs are supported")
        uri = uri[len("s3://") :]
        if not uri:
            raise ValueError("Bucket name must be non-empty")
        split = uri.split("/", maxsplit=1)
        if len(split) == 1:
            bucket = split[0]
            prefix = ""
        else:
            bucket, prefix = split
        if not bucket:
            raise ValueError("Bucket name must be non-empty")
        return bucket, prefix


# TODO: Add support for multiple iterations
class S3IterableDatasetReader(torch.utils.data.IterableDataset):

    def __init__(self, client: MountpointS3Client, source: S3DatasetSource):
        if not(isinstance(client, MountpointS3Client) and isinstance(source, S3DatasetSource)):
            raise ValueError("Expecting (client: MountpointS3Client, source: S3DatasetSource)...")
        self.client = client
        self.source = source

    def __iter__(self):
        for dataset_object in self.source.dataset_objects:
            dataset_object.stream = self.client.get_object(dataset_object.bucket, dataset_object.key)
            yield dataset_object
