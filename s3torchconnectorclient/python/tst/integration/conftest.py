#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import os


def getenv(var: str) -> str:
    v = os.getenv(var)
    if v is None:
        raise Exception(f"required environment variable {var} is not set")
    return v


# TODO: Update with test fixtures
class TestConfig(object):
    """Config object fixture for CI, wrapping region, S3 bucket, S3 express bucket and prefix configuration."""

    region: str
    bucket: str
    express_bucket: str
    express_region: str

    def __init__(
        self, region: str, bucket: str, express_bucket: str, express_region: str
    ):
        self.region = region
        self.bucket = bucket
        self.express_bucket = express_bucket
        self.express_region = express_region


def get_test_config() -> TestConfig:
    """Create a new bucket/prefix fixture for the given test name."""
    region = getenv("CI_REGION")
    bucket = getenv("CI_BUCKET")
    express_bucket = getenv("CI_EXPRESS_BUCKET")
    express_region = getenv("CI_EXPRESS_REGION")

    return TestConfig(region, bucket, express_bucket, express_region)
