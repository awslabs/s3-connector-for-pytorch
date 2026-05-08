#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import sys
import pytest

import s3torchconnector
from mypy import api

if sys.version_info >= (3, 9):
    import importlib.resources


def test_py_typed_exists():
    """Verify py.typed file exists (PEP 561).

    Note existence of py.typed file in source build is only tested in Build Wheels workflow
    where the build artifact itself was tested.
    """
    package_path = importlib.resources.files(s3torchconnector)
    py_typed = package_path / "py.typed"
    assert py_typed.is_file(), "py.typed file not found in s3torchconnector package"


def test_mypy_detects_consumer_code_type_errors(tmp_path):
    """Verify mypy can detect type errors in consumer code using py.typed."""

    test_code = "from s3torchconnector import S3MapDataset\nwrong = S3MapDataset.from_prefix(123, region=456)"
    test_file = tmp_path / "test_mypy_on_consumer_code_incorrect.py"
    test_file.write_text(test_code)

    stdout, stderr, exit_code = api.run([str(test_file)])

    assert exit_code == 1, f"Expected errors, got: {stdout}"
    assert (
        'Argument 1 to "from_prefix" of "S3MapDataset" has incompatible type "int"; expected "str"'
        in stdout
    )
    assert (
        'Argument "region" to "from_prefix" of "S3MapDataset" has incompatible type "int"; expected "str"'
        in stdout
    )


def test_mypy_passes_correct_consumer_code(tmp_path):
    """Verify mypy allows correctly typed consumer code."""

    test_code = "from s3torchconnector import S3MapDataset\nright = S3MapDataset.from_prefix('s3://test-bucket/prefix', region='us-east-1')"
    test_file = tmp_path / "test_mypy_on_consumer_code_correct.py"
    test_file.write_text(test_code)

    stdout, stderr, exit_code = api.run([str(test_file)])

    assert exit_code == 0, f"Expected no errors, got: {stdout}"
    assert "Success: no issues found" in stdout
