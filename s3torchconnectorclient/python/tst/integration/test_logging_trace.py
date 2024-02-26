#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
import io
import sys

PYTHON_TEST_CODE = """
import logging
import os
import sys

os.environ["ENABLE_CRT_LOGS"] = "trace"
from s3torchconnector import S3MapDataset
logging.basicConfig(
    format="%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s"
)

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

def test_logging_trace(image_directory):
    process = subprocess.Popen(["python", "-c", PYTHON_TEST_CODE, image_directory.s3_uri, image_directory.region], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    assert stderr == ''
    [""]
    assert stdout.contains("")
    captured_stdout = io.StringIO()
    sys.stdout = captured_stdout

    try:
        process.wait()
    finally:
        sys.stdout = sys.__stdout__

    assert stdout is not None