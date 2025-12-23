#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import importlib
import warnings
from unittest.mock import patch

import pytest
import s3torchconnector


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
        importlib.reload(s3torchconnector)

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

        importlib.reload(s3torchconnector)

        assert not any(
            "macOS x86_64 wheel support will be deprecated" in str(warning.message)
            for warning in w
        )
