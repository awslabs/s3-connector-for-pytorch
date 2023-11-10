import copyreg

from ._logger_patch import TRACE as LOG_TRACE
from ._logger_patch import _install_trace_logging
from ._s3dataset import S3DatasetException
from .s3object import S3Object

# TODO - Find a better name than `s3dataset_s3_client`

_install_trace_logging()


def _s3dataset_exception_reduce(exc: S3DatasetException):
    return S3DatasetException, exc.args


copyreg.pickle(S3DatasetException, _s3dataset_exception_reduce)

__all__ = [
    "S3Object",
    "LOG_TRACE",
    "S3DatasetException",
]
