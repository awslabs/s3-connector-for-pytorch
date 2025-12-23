#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
from typing import List, Optional
import platform

from ._version import __version__

# https://www.rfc-editor.org/rfc/rfc9110#name-user-agent


class UserAgent:
    def __init__(self, comments: Optional[List[str]] = None):
        if comments is not None and not isinstance(comments, list):
            raise ValueError("Argument comments must be a List[str]")

        os_name = platform.system().lower()
        if os_name == "darwin":
            os_name = "macos"
        os_version = platform.release()
        python_version = platform.python_version()
        arch = platform.machine().lower()
        pytorch_version = self._get_pytorch_version()

        # Build user agent string parts
        parts = [
            f"{__package__}/{__version__}",
            "ua/2.1",
            f"os/{os_name}#{os_version}",
            f"lang/python#{python_version}",
            f"md/arch#{arch}",
            f"md/pytorch#{pytorch_version}",
        ]

        self._user_agent_prefix = " ".join(parts)
        self._comments = comments or []

    @property
    def prefix(self):
        comments_str = "; ".join(filter(None, self._comments))
        if comments_str:
            return f"{self._user_agent_prefix} ({comments_str})"
        return self._user_agent_prefix

    def _get_pytorch_version(self) -> str:
        """Get PyTorch version if imported, else return None."""
        try:
            import torch

            return torch.__version__
        except ImportError:
            return "unknown"
