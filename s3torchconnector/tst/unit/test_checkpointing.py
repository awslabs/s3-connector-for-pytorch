#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
import logging
from io import BytesIO, SEEK_END
from operator import eq
from typing import Any, Callable

import pytest
import torch
from hypothesis import given
from hypothesis.strategies import booleans

from s3torchconnector._s3client import MockS3Client
from s3torchconnector import S3Checkpoint

from _checkpoint_utils import (
    python_primitives,
    byteorders,
    save_with_byteorder,
    load_with_byteorder,
)

TEST_BUCKET = "test-bucket"
TEST_KEY = "test-key"
TEST_REGION = "us-east-1"

use_modern_pytorch_format = booleans()


@given(python_primitives, byteorders, use_modern_pytorch_format)
def test_general_checkpointing_saves_python_primitives(
    data, byteorder, use_modern_pytorch_format
):
    _test_save(data, byteorder, use_modern_pytorch_format)


@given(byteorders, use_modern_pytorch_format)
def test_general_checkpointing_saves_tensor(byteorder, use_modern_pytorch_format):
    tensor = torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])
    _test_save(tensor, byteorder, use_modern_pytorch_format, equal=torch.equal)


@given(byteorders)
def test_general_checkpointing_saves_untyped_storage(byteorder):
    storage = torch.UntypedStorage([1, 2, 3])
    _test_save(
        storage,
        byteorder,
        use_modern_pytorch_format=True,
        equal=lambda a, b: list(a) == list(b),
    )


@pytest.mark.xfail
@given(byteorders)
def test_general_checkpointing_untyped_storage_saves_no_modern_pytorch_format(
    byteorder,
):
    storage = torch.UntypedStorage([1, 2, 3])
    _test_save(
        storage,
        byteorder,
        use_modern_pytorch_format=False,
        equal=lambda a, b: list(a) == list(b),
    )


@given(python_primitives, byteorders, use_modern_pytorch_format)
def test_general_checkpointing_loads_python_primitives(
    data, byteorder, use_modern_pytorch_format
):
    _test_load(data, byteorder, use_modern_pytorch_format)


@given(byteorders, use_modern_pytorch_format)
def test_general_checkpointing_loads_tensor(byteorder, use_modern_pytorch_format):
    tensor = torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])
    _test_save(tensor, byteorder, use_modern_pytorch_format, equal=torch.equal)


@given(byteorders)
def test_general_checkpointing_loads_untyped_storage(byteorder):
    storage = torch.UntypedStorage([1, 2, 3])
    _test_load(
        storage,
        byteorder,
        use_modern_pytorch_format=True,
        equal=lambda a, b: list(a) == list(b),
    )


@pytest.mark.xfail
@given(byteorders)
def test_general_checkpointing_untyped_storage_loads_no_modern_pytorch_format(
    byteorder,
):
    storage = torch.UntypedStorage([1, 2, 3])
    _test_load(
        storage,
        byteorder,
        use_modern_pytorch_format=False,
        equal=lambda a, b: list(a) == list(b),
    )


def test_checkpoint_seek_logging(caplog):
    checkpoint = S3Checkpoint(TEST_REGION)

    # Use MockClient instead of actual client.
    client = MockS3Client(TEST_REGION, TEST_BUCKET)
    checkpoint._client = client

    s3_uri = f"s3://{TEST_BUCKET}/{TEST_KEY}"
    with checkpoint.writer(s3_uri) as writer:
        writer.write(b"test")

    with checkpoint.reader(s3_uri) as reader:
        with caplog.at_level(logging.DEBUG):
            reader.seek(0, SEEK_END)
    assert f"HeadObject {s3_uri}" in caplog.messages


def _test_save(
    data,
    byteorder: str,
    use_modern_pytorch_format: bool,
    *,
    equal: Callable[[Any, Any], bool] = eq,
):
    checkpoint = S3Checkpoint(TEST_REGION)

    # Use MockClient instead of actual client.
    client = MockS3Client(TEST_REGION, TEST_BUCKET)
    checkpoint._client = client

    with checkpoint.writer(f"s3://{TEST_BUCKET}/{TEST_KEY}") as s3_writer:
        save_with_byteorder(data, s3_writer, byteorder, use_modern_pytorch_format)

    serialised = BytesIO(b"".join(client.get_object(TEST_BUCKET, TEST_KEY)))
    assert equal(load_with_byteorder(serialised, byteorder), data)


def _test_load(
    data,
    byteorder: str,
    use_modern_pytorch_format: bool,
    *,
    equal: Callable[[Any, Any], bool] = eq,
):
    checkpoint = S3Checkpoint(TEST_REGION)

    # Put some data to mock bucket and use mock client for Checkpoint
    serialised = BytesIO()
    save_with_byteorder(data, serialised, byteorder, use_modern_pytorch_format)
    serialised.seek(0)

    client = MockS3Client(TEST_REGION, TEST_BUCKET)
    client.add_object(TEST_KEY, serialised.read())
    checkpoint._client = client

    s3reader = checkpoint.reader(f"s3://{TEST_BUCKET}/{TEST_KEY}")

    assert equal(load_with_byteorder(s3reader, byteorder), data)
