#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import pytest
from s3torchconnectorclient import S3Exception

from s3torchconnector._s3client import S3Client

HELLO_WORLD_DATA = b"Hello, World!\n"


def test_no_access_objects_without_profile(empty_directory):
    if empty_directory.profile_bucket is None:
        pytest.skip("No profile bucket configured")

    client = S3Client(
        empty_directory.region,
    )
    filename = f"{empty_directory.prefix}hello_world.txt"

    with pytest.raises(S3Exception):
        put_stream = client.put_object(
            empty_directory.profile_bucket,
            filename,
        )
        put_stream.write(HELLO_WORLD_DATA)
