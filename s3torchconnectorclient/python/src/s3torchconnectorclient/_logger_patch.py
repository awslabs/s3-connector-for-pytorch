#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import logging
import atexit

from ._mountpoint_s3_client import _enable_crt_logging, _disable_logging

TRACE = 5


def _install_trace_logging():
    logging.addLevelName(TRACE, "TRACE")


# Experimental method for enabling verbose logging.
# Please do NOT use unless otherwise instructed.
def _enable_debug_logging():
    atexit.register(_disable_logging)
    _enable_crt_logging()
