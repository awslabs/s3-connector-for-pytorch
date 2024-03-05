#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
from typing import List, Optional

from ._version import __version__

# https://www.rfc-editor.org/rfc/rfc9110#name-user-agent


class UserAgent:
    def __init__(self, comments: Optional[List[str]] = None):
        if comments is not None and not isinstance(comments, list):
            raise ValueError("Argument comments must be a List[str]")
        self._user_agent_prefix = f"{__package__}/{__version__}"
        self._comments = comments or []

    @property
    def prefix(self):
        comments_str = "; ".join(filter(None, self._comments))
        if comments_str:
            return f"{self._user_agent_prefix} ({comments_str})"
        return self._user_agent_prefix
