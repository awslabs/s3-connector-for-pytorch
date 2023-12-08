#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import io
import os
import random

import boto3
import numpy as np
from PIL import Image
import pytest


def getenv(var: str, optional: bool = False) -> str:
    v = os.getenv(var)
    if v is None and not optional:
        raise Exception(f"Required environment variable {var} is not set")
    return v


class BucketPrefixFixture(object):
    """An S3 bucket/prefix and its contents for use in a single unit test. The prefix will be unique
    to this instance, so other concurrent tests won't affect its state."""

    region: str
    bucket: str
    prefix: str
    storage_class: str = None

    def __init__(
        self, region: str, bucket: str, prefix: str, storage_class: str = None
    ):
        self.bucket = bucket
        self.prefix = prefix
        self.region = region
        self.storage_class = storage_class
        self.contents = {}
        session = boto3.Session(region_name=region)
        self.s3 = session.client("s3")

    @property
    def s3_uri(self):
        return f"s3://{self.bucket}/{self.prefix}"

    def add(self, key: str, contents: bytes, **kwargs):
        """Upload an S3 object to this prefix of the bucket."""
        full_key = f"{self.prefix}{key}"
        self.s3.put_object(Bucket=self.bucket, Key=full_key, Body=contents, **kwargs)
        self.contents[full_key] = contents

    def __getitem__(self, index):
        return self.contents[index]

    def __iter__(self):
        return iter(self.contents)


def get_test_bucket_prefix(name: str) -> BucketPrefixFixture:
    """Create a new bucket/prefix fixture for the given test name."""
    bucket = getenv("CI_BUCKET")
    prefix = getenv("CI_PREFIX")
    region = getenv("CI_REGION")
    storage_class = getenv("CI_STORAGE_CLASS", optional=True)
    assert prefix == "" or prefix.endswith("/")

    nonce = random.randrange(2**64)
    prefix = f"{prefix}{name}/{nonce}/"

    return BucketPrefixFixture(region, bucket, prefix, storage_class)


@pytest.fixture
def image_directory(request) -> BucketPrefixFixture:
    """Create a bucket/prefix fixture that contains a directory of random JPG image files."""
    NUM_IMAGES = 10
    IMAGE_SIZE = 100
    fixture = get_test_bucket_prefix(f"{request.node.name}/image_directory")
    for i in range(NUM_IMAGES):
        data = np.random.randint(0, 256, IMAGE_SIZE * IMAGE_SIZE * 3, np.uint8)
        data = data.reshape(IMAGE_SIZE, IMAGE_SIZE, 3)
        image = Image.fromarray(data, "RGB")
        image_bytes = io.BytesIO()
        image.save(image_bytes, "jpeg")
        image_bytes.seek(0)
        image_bytes = image_bytes.read()

        key = f"img{i:03d}.jpg"
        fixture.add(key, image_bytes)

    return fixture


@pytest.fixture
def checkpoint_directory(request) -> BucketPrefixFixture:
    return get_test_bucket_prefix(f"{request.node.name}/checkpoint_directory")
