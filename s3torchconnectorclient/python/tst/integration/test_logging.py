#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
import os.path
import sys
from datetime import datetime

import pytest

PYTHON_TEST_CODE = """
import logging
import os
import sys

os.environ["ENABLE_CRT_LOGS"] = "{0}"

logs_dir_path = "{1}"
if logs_dir_path != "":
    os.environ["CRT_LOGS_DIR_PATH"] = logs_dir_path

from s3torchconnector import S3MapDataset

logging.basicConfig(
    stream=sys.stdout,
    format="%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s",
)
logging.getLogger().setLevel(logging.INFO)

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
            ["DEBUG", "TRACE"],
        ),
        (
            "DEBUG",
            [
                "INFO s3torchconnector.s3map_dataset",
                "DEBUG awscrt::AWSProfile",
                "DEBUG awscrt::AuthCredentialsProvider",
                "DEBUG awscrt::S3Client",
            ],
            ["TRACE"],
        ),
        (
            "TRACE",
            [
                "INFO s3torchconnector.s3map_dataset",
                "DEBUG awscrt::AWSProfile",
                "DEBUG awscrt::AuthCredentialsProvider",
                "TRACE awscrt::event-loop",
                "TRACE awscrt::socket",
                "TRACE awscrt::event-loop",
            ],
            # Python log level is set to INFO in the test script
            ["TRACE s3torchconnector.s3map_dataset"],
        ),
    ],
)
def test_logging_valid(log_level, should_contain, should_not_contain, image_directory):
    out, err = _start_subprocess(log_level, image_directory)
    assert err == ""
    assert all(s in out for s in should_contain)
    assert all(s not in out for s in should_not_contain)


def test_logging_to_file(image_directory):
    dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S.%f")
    log_dir = f"/tmp/test-logs-{dt_string}/"
    out, err = _start_subprocess("INFO", image_directory, log_dir)
    # Standard output contains Python output
    assert err == ""
    assert "INFO s3torchconnector.s3map_dataset" in out
    assert "awscrt" not in out
    # Verifying file logging
    assert os.path.exists(log_dir)
    files = os.listdir(log_dir)
    assert len(files) == 1
    log_file = f"{log_dir}{files[0]}"
    assert os.path.isfile(log_file)
    f = open(log_file, "r")
    file_content = f.read()
    assert all(s in file_content for s in ["awscrt", "INFO"])
    assert all(
        s not in file_content
        for s in ["INFO s3torchconnector.s3map_dataset", "DEBUG", "TRACE"]
    )
    # Cleanup logs file and directory
    os.remove(log_file)
    os.rmdir(log_dir)


def test_logging_off(image_directory):
    out, err = _start_subprocess("off", image_directory)
    assert err == ""
    assert "INFO s3torchconnector.s3map_dataset" in out
    assert "awscrt" not in out


def _start_subprocess(log_level, image_directory, logs_directory=""):
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            PYTHON_TEST_CODE.format(log_level, logs_directory),
            image_directory.s3_uri,
            image_directory.region,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return process.communicate()
