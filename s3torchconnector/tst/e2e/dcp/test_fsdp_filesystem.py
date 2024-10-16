#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from s3torchconnector.dcp.fsdp_filesystem import S3FS


def test_fsdp_filesystem_when_single_thread(checkpoint_directory):
    s3fs = S3FS(checkpoint_directory.region)

    s3fs.init_path()
    pass


def test_fsdp_filesystem_when_multiple_threads(checkpoint_directory):
    s3fs = S3FS(checkpoint_directory.region)
    pass
