#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import pytest

from s3torchconnector.s3reader import (
    S3ReaderConstructor,
    S3ReaderConstructorProtocol,
)

# Shared reader constructors for parametrized tests
# TODO: use this variable in test_distributed_training.py and test_multiprocess_dataloading.py
READER_CONSTRUCTORS = [
    S3ReaderConstructor.sequential(),  # Sequential Reader
    S3ReaderConstructor.range_based(),  # Default range-based reader, with buffer
    S3ReaderConstructor.range_based(buffer_size=0),  # range-based reader, no buffer
]


@pytest.fixture(
    params=READER_CONSTRUCTORS,
    ids=["sequential", "range_based_buffered", "range_based_unbuffered"],
    scope="module",
)
def reader_constructor(request) -> S3ReaderConstructorProtocol:
    """Provide reader constructor (partial(S3Reader)) instances for all supported reader types."""
    return request.param
