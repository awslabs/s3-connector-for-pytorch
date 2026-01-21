#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
import platform
import warnings

import copyreg

from ._logger_patch import TRACE as LOG_TRACE
from ._logger_patch import _install_trace_logging
from ._mountpoint_s3_client import S3Exception, __version__

_install_trace_logging()


def _s3exception_reduce(exc: S3Exception):
    return S3Exception, exc.args


copyreg.pickle(S3Exception, _s3exception_reduce)

__all__ = ["LOG_TRACE", "S3Exception", "__version__"]

# Check for macOS x86_64 and issue deprecation warning
try:
    if platform.system() == "Darwin" and platform.machine() == "x86_64":
        warnings.warn(
            "macOS x86_64 wheel support will be deprecated in a future release. "
            "Please refer to https://github.com/awslabs/s3-connector-for-pytorch/issues/398 for more details.",
            FutureWarning,
            stacklevel=2,
        )
except Exception:
    # Continue if platform detection fails
    pass

# Check for Python 3.8 and issue deprecation warning
try:
    if platform.python_version().startswith("3.8"):
        warnings.warn(
            "Python 3.8 support will be deprecated in a future release. "
            "Please refer to https://github.com/awslabs/s3-connector-for-pytorch/issues/399 for more details.",
            FutureWarning,
            stacklevel=2,
        )
except Exception:
    # Continue if version detection fails
    pass
