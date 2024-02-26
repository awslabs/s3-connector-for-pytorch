#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
import pytest

PYTHON_TEST_CODE = """
import logging
import os
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["ENABLE_CRT_LOGS"] = "{0}"

from s3torchconnector import S3MapDataset

logging.basicConfig(
    stream=sys.stdout,
    format="%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s",
)
logging.getLogger().setLevel(logging.INFO)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        exit("The script needs an S3 uri and a region")
    s3_uri = sys.argv[1]
    region = sys.argv[2]
    map_dataset = S3MapDataset.from_prefix(s3_uri, region=region)
    obj = map_dataset[0]
    assert obj is not None
"""

import subprocess


@pytest.mark.parametrize(
    "log_level, should_contain, should_not_contain",
    [
        (
            "info",
            ["INFO s3torchconnector.s3map_dataset"],
            [
                "DEBUG awscrt::AWSProfile",
                "TRACE awscrt::AWSProfile",
                "DEBUG awscrt::AuthCredentialsProvider",
            ],
        ),
        (
            "debug",
            [
                "INFO s3torchconnector.s3map_dataset",
                "DEBUG awscrt::AWSProfile",
                "DEBUG awscrt::AuthCredentialsProvider",
            ],
            ["TRACE awscrt::AWSProfile"],
        ),
        (
            "trace",
            [
                "INFO s3torchconnector.s3map_dataset",
                "DEBUG awscrt::AWSProfile",
                "DEBUG awscrt::AuthCredentialsProvider",
                "TRACE awscrt::AWSProfile",
            ],
            ["TRACE s3torchconnector.s3map_dataset"],
        ),
    ],
)
def test_logging_valid(log_level, should_contain, should_not_contain, image_directory):
    stdout, stderr = _start_subprocess(log_level, image_directory)
    assert stderr == ""
    assert stdout is not None
    assert all([s in stdout for s in should_contain])
    assert all([s not in stdout for s in should_not_contain])


def test_logging_off(image_directory):
    stdout, stderr = _start_subprocess("off", image_directory)
    assert stderr == ""
    assert stdout is not None
    assert "INFO s3torchconnector.s3map_dataset" in stdout
    assert "awscrt" not in stdout


def test_logging_invalid(image_directory):
    stdout, stderr = _start_subprocess("123", image_directory)
    assert stdout == ""
    assert (
        "s3torchconnectorclient._mountpoint_s3_client.S3Exception: attempted to convert a string that doesn't match an existing log level"
        in stderr
    )


def _start_subprocess(log_level, image_directory):
    process = subprocess.Popen(
        [
            "python",
            "-c",
            PYTHON_TEST_CODE.format(log_level),
            image_directory.s3_uri,
            image_directory.region,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return process.communicate()
