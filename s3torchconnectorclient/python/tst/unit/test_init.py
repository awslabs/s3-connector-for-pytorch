#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import importlib
import warnings
from unittest.mock import patch

import pytest
import s3torchconnectorclient


@pytest.mark.parametrize(
    "system,machine,expect_warn",
    [
        ("Darwin", "x86_64", True),
        ("Linux", "x86_64", False),
        ("Darwin", "arm64", False),
        ("Linux", "aarch64", False),
    ],
)
def test_init_macos_x86_warning(system, machine, expect_warn):
    """Test macOS x86_64 deprecation warning behavior."""
    with patch("platform.system", return_value=system), patch(
        "platform.machine", return_value=machine
    ), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        importlib.reload(s3torchconnectorclient)

        has_warning = any(
            "macOS x86_64 wheel support will be deprecated" in str(warning.message)
            for warning in w
        )
        assert has_warning == expect_warn


def test_init_platform_detection_exception():
    """Test that no crash should happen when platform detection fails."""
    with patch(
        "platform.system", side_effect=OSError("Detection failed")
    ), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        importlib.reload(s3torchconnectorclient)

        assert not any(
            "macOS x86_64 wheel support will be deprecated" in str(warning.message)
            for warning in w
        )


@pytest.mark.parametrize(
    "python_version,expect_warn",
    [
        ("3.8.0", True),
        ("3.8.20", True),
        ("3.9.7", False),
        ("3.13.8", False),
        ("3.13.0t", False),
        ("3.80.0", False),
    ],
)
def test_init_python38_warning(python_version, expect_warn):
    """Test Python 3.8 deprecation warning behavior."""
    with patch(
        "platform.python_version", return_value=python_version
    ), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        importlib.reload(s3torchconnectorclient)

        has_warning = any(
            "Python 3.8 support will be deprecated" in str(warning.message)
            for warning in w
        )
        assert has_warning == expect_warn


def test_init_python_version_detection_exception():
    """Test that no crash should happen when Python version detection fails."""
    with patch(
        "platform.python_version",
        side_effect=OSError("Python version detection failed"),
    ), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        importlib.reload(s3torchconnectorclient)

        assert not any(
            "Python 3.8 support will be deprecated" in str(warning.message)
            for warning in w
        )


def test_init_multiple_warnings():
    """Test both macOS x86_64 and Python 3.8 warnings together."""
    with patch("platform.system", return_value="Darwin"), patch(
        "platform.machine", return_value="x86_64"
    ), patch("platform.python_version", return_value="3.8.20"), warnings.catch_warnings(
        record=True
    ) as w:
        warnings.simplefilter("always")
        importlib.reload(s3torchconnectorclient)

        # Check exactly 2 deprecation warnings
        deprecation_warnings = [warn for warn in w if "deprecated" in str(warn.message)]
        assert len(deprecation_warnings) == 2

        # Test they are FutureWarning with stacklevel 2
        for warning in deprecation_warnings:
            assert warning.category == FutureWarning
            assert warning.filename.endswith("__init__.py")

        # Check specific warning messages
        messages = [str(warn.message) for warn in deprecation_warnings]
        assert any("macOS x86_64" in msg for msg in messages)
        assert any("Python 3.8" in msg for msg in messages)
