#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
from typing import Optional

from ._s3dataset_common import parse_s3_uri
from ._s3client import S3Client, S3ClientConfig
from . import S3Reader, S3Writer


class S3Checkpoint:
    """A checkpoint manager for S3.

    To read a checkpoint from S3, users need to create an S3Reader
    by providing s3_uri of the checkpoint stored in S3. Similarly, to save a
    checkpoint to S3, users need to create an S3Writer by providing s3_uri.
    S3Reader and S3Writer implements io.BufferedIOBase therefore, they can be passed to
    torch.load, and torch.save.
    """

    def __init__(
        self,
        region: str,
        endpoint: Optional[str] = None,
        s3client_config: Optional[S3ClientConfig] = None,
    ):
        self.region = region
        self.endpoint = endpoint
        self._client = S3Client(
            region, endpoint=endpoint, s3client_config=s3client_config
        )

    def reader(self, s3_uri: str) -> S3Reader:
        """Creates an S3Reader from a given s3_uri.

        Args:
            s3_uri (str): A valid s3_uri. (i.e. s3://<BUCKET>/<KEY>)

        Returns:
            S3Reader: a read-only binary stream of the S3 object's contents, specified by the s3_uri.

        Raises:
            S3Exception: An error occurred accessing S3.
        """
        bucket, key = parse_s3_uri(s3_uri)
        return self._client.get_object(bucket, key)

    def writer(self, s3_uri: str) -> S3Writer:
        """Creates an S3Writer from a given s3_uri.

        Args:
            s3_uri (str): A valid s3_uri. (i.e. s3://<BUCKET>/<KEY>)

        Returns:
            S3Writer: a write-only binary stream. The content is saved to S3 using the specified s3_uri.

        Raises:
            S3Exception: An error occurred accessing S3.
        """
        bucket, key = parse_s3_uri(s3_uri)
        return self._client.put_object(bucket, key)
