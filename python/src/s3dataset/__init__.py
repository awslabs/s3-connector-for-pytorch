from ._logger_patch import TRACE as LOG_TRACE
from ._logger_patch import _install_trace_logging
from .s3dataset_base import S3DatasetBase
from .s3iterable_dataset import S3IterableDataset
from .s3mapstyle_dataset import S3MapStyleDataset
from .s3object import S3Object

_install_trace_logging()

__all__ = [
    "LOG_TRACE",
    "S3Object",
    "S3IterableDataset",
    "S3MapStyleDataset",
    "S3DatasetBase",
]
