#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
from __future__ import annotations

from typing import List

import pytest

from s3torchconnector._version import __version__
from s3torchconnector._user_agent import UserAgent

DEFAULT_PREFIX = f"s3torchconnector/{__version__}"


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
