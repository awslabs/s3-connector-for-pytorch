#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import io
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import boto3
import numpy as np
from PIL import Image
import pytest


def getenv(var: str, optional: bool = False) -> str:
    v = os.getenv(var)
    if v is None and not optional:
        raise Exception(f"Required environment variable {var} is not set")
    return v


@dataclass
class BucketPrefixFixture:
    """An S3 bucket/prefix and its contents for use in a single unit test. The prefix will be unique
    to this instance, so other concurrent tests won't affect its state."""

    name: str

    region: str = getenv("CI_REGION")
    bucket: str = getenv("CI_BUCKET")
    prefix: str = getenv("CI_PREFIX")
    storage_class: Optional[str] = getenv("CI_STORAGE_CLASS", optional=True)
    endpoint_url: Optional[str] = getenv("CI_CUSTOM_ENDPOINT_URL", optional=True)
    contents: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.prefix == "" or self.prefix.endswith("/")
        session = boto3.Session(region_name=self.region)
        self.s3 = session.client("s3")

        nonce = random.randrange(2**64)
        self.prefix = f"{self.prefix}{self.name}/{nonce}/"

    @property
    def s3_uri(self):
        return f"s3://{self.bucket}/{self.prefix}"

    def add(self, key: str, contents: bytes, **kwargs):
        """Upload an S3 object to this prefix of the bucket."""
        full_key = f"{self.prefix}{key}"
        self.s3.put_object(Bucket=self.bucket, Key=full_key, Body=contents, **kwargs)
        self.contents[full_key] = contents

    def remove(self, key: str):
        full_key = f"{self.prefix}{key}"
        self.s3.delete_object(Bucket=self.bucket, Key=full_key)

    def __getitem__(self, index):
        return self.contents[index]

    def __iter__(self):
        return iter(self.contents)


@dataclass
class CopyBucketFixture(BucketPrefixFixture):
    src_key: str = "src.txt"
    dst_key: str = "dst.txt"

    @property
    def full_src_key(self):
        return self.prefix + self.src_key

    @property
    def full_dst_key(self):
        return self.prefix + self.dst_key


def get_test_copy_bucket_fixture(name: str) -> CopyBucketFixture:
    copy_bucket_fixture = CopyBucketFixture(name=name)

    # set up / teardown
    copy_bucket_fixture.add(copy_bucket_fixture.src_key, b"Hello, World!\n")
    copy_bucket_fixture.remove(copy_bucket_fixture.dst_key)

    return copy_bucket_fixture


@pytest.fixture
def image_directory(request) -> BucketPrefixFixture:
    """Create a bucket/prefix fixture that contains a directory of random JPG image files."""
    NUM_IMAGES = 10
    IMAGE_SIZE = 100
    fixture = BucketPrefixFixture(f"{request.node.name}/image_directory")
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
def sample_directory(request) -> BucketPrefixFixture:
    fixture = BucketPrefixFixture(f"{request.node.name}/sample_files")
    fixture.add("hello_world.txt", b"Hello, World!\n")
    return fixture


@pytest.fixture
def put_object_tests_directory(request) -> BucketPrefixFixture:
    fixture = BucketPrefixFixture(f"{request.node.name}/put_integration_tests")
    fixture.add("to_overwrite.txt", b"before")
    return fixture


@pytest.fixture
def checkpoint_directory(request) -> BucketPrefixFixture:
    return BucketPrefixFixture(f"{request.node.name}/checkpoint_directory")


@pytest.fixture
def empty_directory(request) -> BucketPrefixFixture:
    return BucketPrefixFixture(f"{request.node.name}/empty_directory")


@pytest.fixture
def copy_directory(request) -> CopyBucketFixture:
    return get_test_copy_bucket_fixture(f"{request.node.name}/copy_directory")
