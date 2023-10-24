from typing import Iterable, Union, Tuple
from s3dataset.s3object import S3Object
from s3dataset._s3dataset import MountpointS3Client


"""
s3dataset_base.py
    Base class for S3 datasets, containing logic for URIs parsing and objects listing. 
"""


class S3DatasetBase:
    def __init__(
        self,
        client: MountpointS3Client,
        dataset_objects: Iterable[S3Object] = (),
    ):
        self._client = client
        self.dataset_objects = dataset_objects

    @property
    def region(self):
        return self._client.region

    @classmethod
    def from_objects(
        cls,
        object_uris: Union[str, Iterable[str]],
        *,
        region: str = None,
        client: MountpointS3Client = None,
    ):
        """
        Creates an S3IterableDataset from the object(s) URI(s) passed as parameters.
        Args:
          object_uris(str or Iterable[str]):
            S3 URI of the object(s) desired.
          region(str or None):
            The S3 region where the objects are stored.
            If this is provided a MountpointS3Client will be instantiated with the region.
          client:
            MountpointS3Client instance to be used for S3 interactions.
        """
        cls._validate_arguments(region, client)
        if isinstance(object_uris, str):
            object_uris = [object_uris]
        # TODO: We should be consistent with URIs parsing. Revise if we want to do this upfront or lazily.
        bucket_key_pairs = [_parse_s3_uri(uri) for uri in object_uris]
        client = client or MountpointS3Client(region)
        return cls(client, cls._bucket_keys_to_s3objects(client, bucket_key_pairs))

    @classmethod
    def from_bucket(
        cls,
        bucket: str,
        prefix: str = None,
        *,
        region: str = None,
        client: MountpointS3Client = None,
    ):
        """
        Creates an S3IterableDataset with the objects found in S3 under bucket/prefix.
        Args:
          bucket(str):
            Name of the S3 bucket where the objects are stored.
          prefix(str or None):
            The S3 prefix for the objects in scope.
          region(str or None):
            The S3 region where the bucket is.
            If this is provided a MountpointS3Client will be instantiated with the region.
          client:
            MountpointS3Client instance to be used for S3 interactions.
        """
        cls._validate_arguments(region, client)
        client = client or MountpointS3Client(region)
        return cls(client, cls._list_objects_for_bucket(client, bucket, prefix))

    @staticmethod
    def _bucket_keys_to_s3objects(
        client: MountpointS3Client, bucket_key_pairs: Iterable[Tuple[str, str]]
    ) -> Iterable[S3Object]:
        for bucket, key in bucket_key_pairs:
            yield S3Object(bucket, key, stream=client.get_object(bucket, key))

    @staticmethod
    def _list_objects_for_bucket(
        client: MountpointS3Client, bucket: str, prefix: str = None
    ) -> Iterable[S3Object]:
        # TODO: Test if it works with more than 1000 objs (perhaps set a lower page size in MockClient)
        list_object_stream = client.list_objects(bucket, prefix or "")

        for page in list_object_stream:
            for object_info in page.object_info:
                yield S3Object(
                    bucket,
                    object_info.key,
                    object_info,
                    client.get_object(bucket, object_info.key),
                )

    @staticmethod
    def _validate_arguments(region: str = None, client: MountpointS3Client = None):
        if not region and not client:
            raise ValueError("Either region or client must be valid.")
        if region and client:
            raise ValueError("Only one of region / client should be passed.")


# TODO: Check boto3 implementation for this
def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    # TODO: We should be able to support more through Mountpoint, not sure if we want to
    if not uri or not uri.startswith("s3://"):
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
