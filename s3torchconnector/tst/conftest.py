#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import pytest

from s3torchconnector.s3reader import (
    S3ReaderConstructor,
    S3ReaderConstructorProtocol,
    SequentialS3Reader,
    RangedS3Reader,
    DCPOptimizedS3Reader,
)

READER_TYPE_STRING_TO_CLASS = {
    "sequential": SequentialS3Reader,
    "range_based": RangedS3Reader,
    "dcp_optimized": DCPOptimizedS3Reader,
}

# Shared reader constructors for parametrized tests
# TODO: use this variable in test_distributed_training.py and test_multiprocess_dataloading.py
READER_CONSTRUCTORS = [
    ("sequential", S3ReaderConstructor.sequential()),
    ("range_based_with_buffer", S3ReaderConstructor.range_based()),
    ("range_based_no_buffer", S3ReaderConstructor.range_based(buffer_size=0)),
]

# Include dcp_optimized for DCP tests
DCP_READER_CONSTRUCTORS = READER_CONSTRUCTORS + [
    ("dcp_optimized", S3ReaderConstructor.dcp_optimized()),
]


@pytest.fixture(
    params=[constructor for _, constructor in READER_CONSTRUCTORS],
    ids=[name for name, _ in READER_CONSTRUCTORS],
    scope="module",
)
def reader_constructor(request) -> S3ReaderConstructorProtocol:
    """Provide reader constructor (partial(S3Reader)) instances for all supported reader types."""
    return request.param


@pytest.fixture(
    params=[constructor for _, constructor in DCP_READER_CONSTRUCTORS],
    ids=[name for name, _ in DCP_READER_CONSTRUCTORS],
    scope="module",
)
def dcp_reader_constructor(request) -> S3ReaderConstructorProtocol:
    """Provide reader constructor instances for DCP tests including dcp_optimized."""
    return request.param
