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


class BucketPrefixData(object):
    """An S3 bucket/prefix and its contents for use in a single unit test. The prefix will be unique
    to this instance, so other concurrent tests won't affect its state."""

    region: str
    bucket: str
    prefix: str
    storage_class: str = None
    contents: dict
    profile_arn: str = None
    profile_bucket: str = None

    def __init__(
        self,
        region: str,
        bucket: str,
        prefix: str,
        storage_class: str = None,
        contents: dict = None,
        profile_arn: str = None,
        profile_bucket: str = None,
    ):
        self.bucket = bucket
        self.prefix = prefix
        self.region = region
        self.storage_class = storage_class
        self.contents = contents or {}
        self.profile_arn = profile_arn
        self.profile_bucket = profile_bucket

    @property
    def s3_uri(self):
        return f"s3://{self.bucket}/{self.prefix}"

    def is_express_storage(self):
        return self.storage_class == "EXPRESS_ONEZONE"

    def __getitem__(self, index):
        return self.contents[index]

    def __iter__(self):
        return iter(self.contents)


class BucketPrefixFixture(BucketPrefixData):
    """An S3 bucket/prefix and its contents for use in a single unit test. The prefix will be unique
    to this instance, so other concurrent tests won't affect its state."""

    def __init__(
        self,
        region: str,
        bucket: str,
        prefix: str,
        storage_class: str = None,
        profile_arn: str = None,
        profile_bucket: str = None,
    ):
        super().__init__(
            region,
            bucket,
            prefix,
            storage_class,
            profile_arn=profile_arn,
            profile_bucket=profile_bucket,
        )
        session = boto3.Session(region_name=region)
        self.s3 = session.client("s3")

    def add(self, key: str, contents: bytes, **kwargs):
        """Upload an S3 object to this prefix of the bucket."""
        full_key = f"{self.prefix}{key}"
        self.s3.put_object(Bucket=self.bucket, Key=full_key, Body=contents, **kwargs)
        self.contents[full_key] = contents

    def get_data_snapshot(self):
        """Returns a read-only copy of the current instance's data.

        The returned object cannot modify the actual S3 bucket.
        Useful when passing data to another process without serializing s3 client
        """
        return BucketPrefixData(
            self.region,
            self.bucket,
            self.prefix,
            self.storage_class,
            self.contents,
            self.profile_arn,
            self.profile_bucket,
        )


def get_test_bucket_prefix(name: str) -> BucketPrefixFixture:
    """Create a new bucket/prefix fixture for the given test name."""
    bucket = getenv("CI_BUCKET")
    prefix = getenv("CI_PREFIX")
    region = getenv("CI_REGION")
    storage_class = getenv("CI_STORAGE_CLASS", optional=True)
    profile_arn = getenv("CI_PROFILE_ROLE", optional=True)
    profile_bucket = getenv("CI_PROFILE_BUCKET", optional=True)
    assert prefix == "" or prefix.endswith("/")

    nonce = random.randrange(2**64)
    prefix = f"{prefix}{name}/{nonce}/"

    return BucketPrefixFixture(
        region, bucket, prefix, storage_class, profile_arn, profile_bucket
    )


@pytest.fixture
def image_directory(request) -> BucketPrefixFixture:
    NUM_IMAGES = 10
    IMAGE_SIZE = 100
    return _create_image_directory_fixture(NUM_IMAGES, IMAGE_SIZE, request.node.name)


@pytest.fixture
def image_directory_for_dp(request) -> BucketPrefixFixture:
    """When conducting distributed training tests, be cautious about the number of files (images) in the test dataset.
    If the total number of images cannot be evenly divided by the number of workers,
    the DistributedSampler will duplicate a subset of the images across workers to ensure an equal
    distribution of data among all processes. This duplication of images will cause
    integration distributed training test to fail.
    """
    NUM_IMAGES = 36
    IMAGE_SIZE = 100
    return _create_image_directory_fixture(NUM_IMAGES, IMAGE_SIZE, request.node.name)


def _create_image_directory_fixture(num_image: int, image_size: int, node_name: str):
    """Create a bucket/prefix fixture that contains a directory of random JPG image files."""
    fixture = get_test_bucket_prefix(f"{node_name}/image_directory")
    for i in range(num_image):
        data = np.random.randint(0, 256, image_size * image_size * 3, np.uint8)
        data = data.reshape(image_size, image_size, 3)
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


@pytest.fixture
def empty_directory(request) -> BucketPrefixFixture:
    return get_test_bucket_prefix(f"{request.node.name}/empty_directory")
