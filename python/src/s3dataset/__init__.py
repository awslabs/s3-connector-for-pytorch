from s3dataset._logger_patch import TRACE as LOG_TRACE
from s3dataset._logger_patch import _install_trace_logging

_install_trace_logging()

__all__ = [
    "LOG_TRACE",
]
