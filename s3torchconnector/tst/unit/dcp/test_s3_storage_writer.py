#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import pytest
from s3torchconnector.dcp import S3StorageWriter

TEST_REGION = "eu-east-1"
TEST_BUCKET = "test-bucket"
TEST_KEY = "test-key.txt"
TEST_PATH = f"s3://{TEST_BUCKET}/{TEST_KEY}"

@pytest.mark.parametrize("thread_count", [1, 2, 4, 8, 16])
def test_s3storage_writer_thread_count(thread_count):
    storage_writer = S3StorageWriter(region=TEST_REGION, path=TEST_PATH, thread_count=thread_count)
    assert storage_writer.thread_count == thread_count

def test_s3storage_writer_thread_count_none_defaults_to_one():
    storage_writer = S3StorageWriter(region=TEST_REGION, path=TEST_PATH)
    assert storage_writer.thread_count == 1