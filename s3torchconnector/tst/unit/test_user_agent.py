#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
from __future__ import annotations

from typing import List
import platform
from unittest.mock import MagicMock, patch

import pytest
import torch

from s3torchconnector._version import __version__
from s3torchconnector._user_agent import UserAgent

# User Agent Default Prefix
PYTHON_VERSION = platform.python_version()
OS_NAME = platform.system().lower()
if OS_NAME == "darwin":
    OS_NAME = "macos"
OS_VERSION = platform.release()
ARCH = platform.machine().lower()
PYTORCH_VERSION = torch.__version__
DEFAULT_PREFIX = f"s3torchconnector/{__version__} ua/2.1 os/{OS_NAME}#{OS_VERSION} lang/python#{PYTHON_VERSION} md/arch#{ARCH} md/pytorch#{PYTORCH_VERSION}"


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


def test_user_agent_os_darwin_to_macos():
    """Test Darwin OS gets converted to macos."""
    with patch("platform.system", return_value="Darwin"), patch(
        "platform.release", return_value="20.0.0"
    ):
        user_agent = UserAgent()
        assert "os/macos#20.0.0" in user_agent.prefix


def test_user_agent_pytorch_unavailable():
    """Test PyTorch version when unavailable (only patches method)."""
    with patch.object(UserAgent, "_get_pytorch_version", return_value="unknown"):
        user_agent = UserAgent()
        assert "md/pytorch#unknown" in user_agent.prefix


@pytest.mark.parametrize("invalid_comment", [0, "string"])
def test_invalid_comments_argument(invalid_comment):
    with pytest.raises(ValueError, match=r"Argument comments must be a List\[str\]"):
        UserAgent(invalid_comment)
