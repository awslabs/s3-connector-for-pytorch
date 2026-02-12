#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
from __future__ import annotations

import re
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


def test_user_agent_get_default_prefix_format():
    """Test get_default_prefix returns the expected format structure."""
    prefix = UserAgent.get_default_prefix()

    # Example: s3torchconnector/1.4.3 ua/2.1 os/macos#20.0.0 lang/python#3.12.9 md/arch#arm64 md/pytorch#2.8.0
    pattern = (
        r"^s3torchconnector/\d+\.\d+\.\d+ "
        r"ua/2\.1 "
        r"os/[a-z]+#\S+ "  # version can vary, e.g. 5.15.5-generic
        r"lang/python#\d+\.\d+\.\d+[a-z]* "  # Allow e.g. 3.14t
        r"md/arch#[a-z0-9_]+ "  # expect x86_64, arm64, or aarch64
        r"md/pytorch#\S+$"  # varies, e.g. "2.9.1+cu121", or "torch==2.11.0.dev20260121"
    )
    assert re.match(pattern, prefix), f"Unexpected user agent format: {prefix}"


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


def test_user_agent_pytorch_version():
    """Test PyTorch version is included correctly."""
    with patch("torch.__version__", "2.10.0"):
        user_agent = UserAgent()
        assert "md/pytorch#2.10.0" in user_agent.prefix


@pytest.mark.parametrize("invalid_comment", [0, "string"])
def test_invalid_comments_argument(invalid_comment):
    with pytest.raises(ValueError, match="Argument comments must be a List\[str\]"):
        UserAgent(invalid_comment)
