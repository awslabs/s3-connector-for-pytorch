#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import copyreg

from ._logger_patch import TRACE as LOG_TRACE
from ._logger_patch import _install_trace_logging
from ._mountpoint_s3_client import S3Exception, __version__

_install_trace_logging()


def _s3exception_reduce(exc: S3Exception):
    return S3Exception, exc.args


copyreg.pickle(S3Exception, _s3exception_reduce)

__all__ = ["LOG_TRACE", "S3Exception", "__version__"]
