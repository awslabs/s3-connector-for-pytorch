#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import io
from typing import Callable, Optional, Union, cast

from s3torchconnectorclient._mountpoint_s3_client import (
    ObjectInfo,
    GetObjectStream,
    HeadObjectResult,
)
from .s3reader_config import S3ReaderConfig
from .sequential import SequentialS3Reader
from .ranged import RangedS3Reader


class S3Reader(io.BufferedIOBase):
    """A read-only, file like representation of a single object stored in S3.

    This class acts as a factory that returns either a sequential or range-based reader
    implementation based on the provided configuration.

    Args:
        bucket (str): S3 bucket name
        key (str): Object key in the bucket
        get_object_info: Callable that returns object metadata
        get_stream: Callable that returns object data stream
        reader_config (S3ReaderConfig, optional): Configuration for reader behavior.
    """

    def __new__(
        cls,
        bucket: str,
        key: str,
        get_object_info: Callable[[], Union[ObjectInfo, HeadObjectResult]],
        get_stream: Union[
            Callable[[], GetObjectStream],
            Callable[[Optional[int], Optional[int]], GetObjectStream],
        ],
        reader_config: Optional[S3ReaderConfig] = None,
    ):
        if reader_config is not None and not isinstance(reader_config, S3ReaderConfig):
            raise TypeError(
                f"reader_config must be an instance of S3ReaderConfig, got {type(reader_config)}"
            )

        config = reader_config or S3ReaderConfig()

        if config.reader_type == S3ReaderConfig.ReaderType.SEQUENTIAL:
            return SequentialS3Reader(
                bucket,
                key,
                get_object_info,
                cast(Callable[[], GetObjectStream], get_stream),
            )
        elif config.reader_type == S3ReaderConfig.ReaderType.RANGE_BASED:
            return RangedS3Reader(
                bucket,
                key,
                get_object_info,
                cast(
                    Callable[[Optional[int], Optional[int]], GetObjectStream],
                    get_stream,
                ),
            )

        raise ValueError(f"Unsupported reader type: {config.reader_type}")
