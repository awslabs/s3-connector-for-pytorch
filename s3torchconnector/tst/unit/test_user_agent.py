#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
from __future__ import annotations

from typing import List
from unittest.mock import patch

import pytest

from s3torchconnector._version import __version__
from s3torchconnector._user_agent import UserAgent

DEFAULT_PREFIX = UserAgent.get_default_prefix()


@pytest.mark.parametrize(
    "comments, expected_prefix",
    [
        (None, DEFAULT_PREFIX),
        ([], DEFAULT_PREFIX),
        ([""], DEFAULT_PREFIX),
        (["", ""], DEFAULT_PREFIX),
        (
            ["component/version", "metadata"],
            f"{DEFAULT_PREFIX} (component/version; metadata)",
        ),
    ],
)
def test_user_agent_creation(comments: List[str] | None, expected_prefix: str):
    user_agent = UserAgent(comments)
    assert user_agent.prefix == expected_prefix


def test_default_user_agent_creation():
    user_agent = UserAgent()
    assert user_agent.prefix == DEFAULT_PREFIX


def test_user_agent_format_structure():
    """Test that all expected fields appears in the user agent."""
    user_agent = UserAgent()
    parts = user_agent.prefix.split()

    assert len(parts) == 6
    assert parts[0].startswith("s3torchconnector/")
    assert parts[1] == "ua/2.1"
    assert parts[2].startswith("os/")
    assert parts[3].startswith("lang/python#")
    assert parts[4].startswith("md/arch#")
    assert parts[5].startswith("md/pytorch#")


@pytest.mark.parametrize(
    "os_name,expected",
    [
        ("Linux", "linux"),
        ("Darwin", "macos"),  # Darwin maps to macos
    ],
)
def test_user_agent_os_mapping(os_name, expected):
    """Test OS name mapping and version inclusion."""
    with patch("platform.system", return_value=os_name), patch(
        "platform.release", return_value="1.2.3"
    ):
        user_agent = UserAgent()
        assert f"os/{expected}#1.2.3" in user_agent.prefix


def test_user_agent_python_version():
    """Test Python version is included correctly."""
    with patch("platform.python_version", return_value="3.12.9"):
        user_agent = UserAgent()
        assert "lang/python#3.12.9" in user_agent.prefix


@pytest.mark.parametrize(
    "arch_input,expected",
    [
        ("x86_64", "x86_64"),
        ("X86_64", "x86_64"),  # Test case conversion
        ("aarch64", "aarch64"),
        ("arm64", "arm64"),
    ],
)
def test_user_agent_architecture_field(arch_input, expected):
    """Test architecture field is included and lowercased."""
    with patch("platform.machine", return_value=arch_input):
        user_agent = UserAgent()
        assert f"md/arch#{expected}" in user_agent.prefix


def test_get_pytorch_version_available():
    """Test _get_pytorch_version when torch is available."""
    version = UserAgent._get_pytorch_version()
    assert version != "unknown"
    assert isinstance(version, str)


def test_get_pytorch_version_unavailable():
    """Test _get_pytorch_version when torch is not imported."""
    with patch.dict("sys.modules", {"torch": None}):
        version = UserAgent._get_pytorch_version()
        assert version == "unknown"


@pytest.mark.parametrize("invalid_comment", [0, "string"])
def test_invalid_comments_argument(invalid_comment):
    with pytest.raises(ValueError, match=r"Argument comments must be a List\[str\]"):
        UserAgent(invalid_comment)
