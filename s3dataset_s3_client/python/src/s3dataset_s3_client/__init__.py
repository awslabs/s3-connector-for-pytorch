from ._logger_patch import TRACE as LOG_TRACE
from ._logger_patch import _install_trace_logging
from .s3object import S3Object

_install_trace_logging()


__all__ = ["S3Object", "LOG_TRACE"]
