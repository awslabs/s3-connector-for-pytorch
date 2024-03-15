#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import logging

TRACE = 5


def _install_trace_logging() -> None:
    logging.addLevelName(TRACE, "TRACE")
