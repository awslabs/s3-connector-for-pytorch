#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
import os.path
import platform
import sys
import tempfile
from typing import List

import pytest

from conftest import BucketPrefixFixture

PYTHON_TEST_CODE = """
import logging
import os
import sys

s3_uri = sys.argv[1]
region = sys.argv[2]
debug_logs_config = sys.argv[3]
logs_dir_path = sys.argv[4]

if debug_logs_config != "":
    os.environ["S3_TORCH_CONNECTOR_DEBUG_LOGS"] = debug_logs_config
if logs_dir_path != "":
    os.environ["S3_TORCH_CONNECTOR_LOGS_DIR_PATH"] = logs_dir_path

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
    "debug_logs_config, should_contain, should_not_contain",
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
def test_crt_logging(
    debug_logs_config: str,
    should_contain: List[str],
    should_not_contain: List[str],
    image_directory: BucketPrefixFixture,
):
    out, err = _start_subprocess(image_directory, debug_logs_config=debug_logs_config)
    assert err == ""
    assert all(s in out for s in should_contain)
    assert all(s not in out for s in should_not_contain)


def test_default_logging_env_filters_unset(image_directory: BucketPrefixFixture):
    out, err = _start_subprocess(image_directory)
    # Standard output contains Python output
    assert err == ""
    assert "INFO s3torchconnector.s3map_dataset" in out
    assert "awscrt::" not in out


def test_logging_to_file_env_filters_unset(image_directory: BucketPrefixFixture):
    with tempfile.TemporaryDirectory() as log_dir:
        print("Created temporary directory", log_dir)
        out, err = _start_subprocess(image_directory, logs_directory=log_dir)
        # Standard output contains Python output
        assert err == ""
        assert "INFO s3torchconnector.s3map_dataset" in out
        assert "awscrt" not in out
        files = os.listdir(log_dir)
        assert len(files) == 0


@pytest.mark.parametrize(
    "debug_logs_config, out_should_contain, out_should_not_contain, file_should_contain, file_should_not_contain",
    [
        (
            "info",
            ["INFO s3torchconnector.s3map_dataset"],
            ["awscrt"],
            ["INFO awscrt::", "INFO"],
            ["INFO s3torchconnector.s3map_dataset", "DEBUG", "TRACE"],
        ),
        (
            "debug,awscrt=off",
            ["INFO s3torchconnector.s3map_dataset"],
            ["awscrt::"],
            ["DEBUG", "mountpoint_s3_client"],
            ["awscrt::", "INFO s3torchconnector.s3map_dataset", "TRACE"],
        ),
        (
            "debug",
            ["INFO s3torchconnector.s3map_dataset"],
            ["awscrt"],
            ["DEBUG", "mountpoint_s3_client", "DEBUG awscrt::", "INFO awscrt::"],
            ["INFO s3torchconnector.s3map_dataset", "TRACE awscrt"],
        ),
    ],
)
def test_logging_to_file(
    debug_logs_config: str,
    out_should_contain: List[str],
    out_should_not_contain: List[str],
    file_should_contain: List[str],
    file_should_not_contain: List[str],
    image_directory: BucketPrefixFixture,
):
    with tempfile.TemporaryDirectory() as log_dir:
        print("Created temporary directory", log_dir)
        out, err = _start_subprocess(
            image_directory,
            debug_logs_config=debug_logs_config,
            logs_directory=log_dir,
        )
        # Standard output contains Python output
        assert err == ""
        assert all(s in out for s in out_should_contain)
        assert all(s not in out for s in out_should_not_contain)
        files = os.listdir(log_dir)
        # There will be two files if the hour changes while running the test
        assert len(files) >= 1
        log_files = [os.path.join(log_dir, f) for f in files]
        assert all(os.path.isfile(log_file) for log_file in log_files)
        log_files_content = "".join(
            [_read_log_file(log_file) for log_file in log_files]
        )
        assert all(s in log_files_content for s in file_should_contain)
        assert all(s not in log_files_content for s in file_should_not_contain)


@pytest.mark.xfail(
    reason="tracing-subscriber 0.3.20 EnvFilter parsing regression - see tokio-rs/tracing#3371"
)
def test_invalid_logging(image_directory):
    out, err = _start_subprocess(image_directory, debug_logs_config="invalid123.&/?")
    assert (
        "s3torchconnectorclient._mountpoint_s3_client.S3Exception: invalid filter directive"
        in err
    )


def _start_subprocess(
    image_directory: BucketPrefixFixture,
    *,
    debug_logs_config: str = "",
    logs_directory: str = "",
):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            PYTHON_TEST_CODE,
            image_directory.s3_uri,
            image_directory.region,
            debug_logs_config,
            logs_directory,
        ],
        capture_output=True,
        text=True,
    )
    return result.stdout, result.stderr


def _read_log_file(log_file: str):
    with open(log_file) as f:
        return f.read()
