#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
import os.path
import sys
import tempfile

import pytest

PYTHON_TEST_CODE = """
import logging
import os
import sys

s3_uri = sys.argv[1]
region = sys.argv[2]
os.environ["ENABLE_CRT_LOGS"] = sys.argv[3]
logs_dir_path = sys.argv[4]
if logs_dir_path != "":
    os.environ["CRT_LOGS_DIR_PATH"] = logs_dir_path

from s3torchconnector import S3MapDataset

logging.basicConfig(
    stream=sys.stdout,
    format="%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s",
)
logging.getLogger().setLevel(logging.INFO)

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
        ("OFF", ["INFO s3torchconnector.s3map_dataset"], ["awscrt"]),
    ],
)
def test_logging_valid(log_level, should_contain, should_not_contain, image_directory):
    out, err = _start_subprocess(log_level, image_directory)
    assert err == ""
    assert all(s in out for s in should_contain)
    assert all(s not in out for s in should_not_contain)


def test_logging_to_file(image_directory):
    with tempfile.TemporaryDirectory() as log_dir:
        print("Created temporary directory", log_dir)
        out, err = _start_subprocess("INFO", image_directory, log_dir)
        # Standard output contains Python output
        assert err == ""
        assert "INFO s3torchconnector.s3map_dataset" in out
        assert "awscrt" not in out
        files = os.listdir(log_dir)
        assert len(files) == 1
        log_file = os.path.join(log_dir, files[0])
        assert os.path.isfile(log_file)
        with open(log_file) as f:
            file_content = f.read()
            assert all(s in file_content for s in ["awscrt", "INFO"])
            assert all(
                s not in file_content
                for s in ["INFO s3torchconnector.s3map_dataset", "DEBUG", "TRACE"]
            )


def _start_subprocess(log_level, image_directory, logs_directory=""):
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            PYTHON_TEST_CODE,
            image_directory.s3_uri,
            image_directory.region,
            log_level,
            logs_directory,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return process.communicate()
