from typing import Iterable, Union
from s3dataset.s3object import S3Object
from s3dataset._s3dataset import MountpointS3Client


class S3DatasetBase:
    def __init__(self, dataset_objects: Iterable[S3Object] = ()):
        self.dataset_objects = dataset_objects

    @classmethod
    def from_objects(
        cls,
        object_uri: Union[str, Iterable[str]],
        *,
        region: str = None,
        client: MountpointS3Client = None,
    ):
        s3client = cls._pick_client(region, client)
        return cls(cls._objects_to_s3objects(s3client, object_uri))

    @classmethod
    def from_bucket(
        cls,
        bucket: str,
        prefix: str = None,
        *,
        region: str = None,
        client: MountpointS3Client = None,
    ):
        s3client = cls._pick_client(region, client)
        return cls(cls._list_objects_for_bucket(s3client, bucket, prefix))

    def _objects_to_s3objects(
        client: MountpointS3Client, object_uris: Iterable[str]
    ) -> Iterable[S3Object]:
        for uri in object_uris:
            bucket, key = _parse_s3_uri(uri)
            yield S3Object(bucket, key, stream=client.get_object(bucket, key))

    def _list_objects_for_bucket(
        client: MountpointS3Client, bucket: str, prefix: str = None
    ) -> Iterable[S3Object]:
        # TODO: Test if it works with more than 1000 objs (perhaps set a lower page size in MockClient)
        if prefix:
            stream = client.list_objects(bucket, prefix)
        else:
            stream = client.list_objects(bucket)

        for page in stream:
            for object_info in page.object_info:
                yield S3Object(
                    bucket,
                    object_info.key,
                    object_info,
                    client.get_object(bucket, object_info.key),
                )

    def _pick_client(
        region: str = None, client: MountpointS3Client = None
    ) -> MountpointS3Client:
        if not region and not client:
            raise ValueError("Either region or client must be valid.")
        if region and client:
            raise ValueError("Only one of region / client should be passed.")
        if client:
            return client

        return MountpointS3Client(region)


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
