import io
from typing import Iterable

import torch
from s3dataset._s3dataset import MountpointS3Client, ObjectInfo, GetObjectStream


class S3DatasetObject(io.BufferedIOBase):
    def __init__(self, bucket: str, key: str, object_info: ObjectInfo = None, stream: GetObjectStream = None):
        super().__init__()
        self.bucket = bucket
        self.key = key
        self.object_info = object_info
        self.stream = stream

    # TODO: Support multiple sizes
    def read(self, size=-1):
        return b''.join(self.stream)
        pass

class S3DatasetSource():

    def __init__(self, dataset_objects: Iterable[S3DatasetObject] = ()):
        self.dataset_objects = dataset_objects

    @classmethod
    def from_object_uris(cls, client: MountpointS3Client, object_uris: Iterable[str]):
        return cls._object_uris_to_dataset_objects(client, object_uris)

    @classmethod
    def from_bucket(cls, client: MountpointS3Client, bucket: str, prefix: str = None):
        return cls._list_objects_for_bucket(client, bucket, prefix)

    @staticmethod
    def _object_uris_to_dataset_objects(client: MountpointS3Client, object_uris: Iterable[str]):
        for uri in object_uris:
            bucket, key = S3DatasetSource._parse_s3_uri(uri)
            yield S3DatasetObject(bucket, key, stream = client.get_object(bucket, key))

    @staticmethod
    def _list_objects_for_bucket(client: MountpointS3Client, bucket: str, prefix: str = None) -> Iterable[S3DatasetObject]:
        if prefix is None:
            stream = client.list_objects(bucket)
        else:
            stream = client.list_objects(bucket, prefix)
        return (S3DatasetObject(bucket, object_info.key, object_info, client.get_object(bucket, object_info.key)) for page in stream for object_info in page.object_info)

    @staticmethod
    def _parse_s3_uri(uri: str) -> [str, str]:
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
class S3IterableDataset(torch.utils.data.IterableDataset):

    def __init__(self, client: MountpointS3Client, source: S3DatasetSource):
        # TODO: find how to check isinstance(source, S3DatasetSource) this to match also Generator
        if not(isinstance(client, MountpointS3Client)):
            raise ValueError("Expecting (client: MountpointS3Client, source: S3DatasetSource)...")
        self.client = client
        self.source = source

    def __iter__(self):
        yield from self.source
