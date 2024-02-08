#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from io import BytesIO
from operator import eq
from pathlib import Path
from typing import Callable, Any

import pytest
import torch
from hypothesis import given

from s3torchconnector._s3client import MockS3Client
from s3torchconnector.lightning import S3LightningCheckpoint
from .._checkpoint_utils import (
    python_primitives,
    byteorders,
    save_with_byteorder,
    load_with_byteorder,
    _patch_byteorder,
)
from s3torchconnectorclient import S3Exception

TEST_BUCKET = "test-bucket"
TEST_KEY = "test-key"
TEST_REGION = "us-east-1"


@given(python_primitives, byteorders)
def test_lightning_checkpointing_saves_python_primitives(data, byteorder):
    _test_save(data, byteorder)


@given(byteorders)
def test_lightning_checkpointing_saves_tensor(byteorder):
    tensor = torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])
    _test_save(tensor, byteorder, equal=torch.equal)


@given(byteorders)
def test_lightning_checkpointing_saves_untyped_storage(byteorder):
    storage = torch.UntypedStorage([1, 2, 3])
    _test_save(
        storage,
        byteorder,
        equal=lambda a, b: list(a) == list(b),
    )


@given(python_primitives, byteorders)
def test_lightning_checkpointing_loads_python_primitives(data, byteorder):
    _test_load(data, byteorder)


@given(byteorders)
def test_lightning_checkpointing_loads_tensor(byteorder):
    tensor = torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])
    _test_load(tensor, byteorder, equal=torch.equal)


@given(byteorders)
def test_lightning_checkpointing_loads_untyped_storage(byteorder):
    storage = torch.UntypedStorage([1, 2, 3])
    _test_load(
        storage,
        byteorder,
        equal=lambda a, b: list(a) == list(b),
    )


def test_removes_checkpoint():
    client = MockS3Client(TEST_REGION, TEST_BUCKET)
    client.add_object(TEST_KEY, b"data")

    s3_lightning_checkpoint = S3LightningCheckpoint(TEST_REGION)
    s3_lightning_checkpoint._client = client
    s3_lightning_checkpoint.remove_checkpoint(f"s3://{TEST_BUCKET}/{TEST_KEY}")

    with pytest.raises(S3Exception) as error:
        client.get_object(TEST_BUCKET, TEST_KEY).read()
    assert str(error.value) == "Service error: The key does not exist"


@pytest.mark.parametrize(
    "checkpoint_method_name, kwargs",
    [
        ("save_checkpoint", {"path": Path(), "checkpoint": None}),
        ("load_checkpoint", {"path": Path()}),
        ("remove_checkpoint", {"path": Path()}),
    ],
)
def test_invalid_path(checkpoint_method_name, kwargs):
    s3_lightning_checkpoint = S3LightningCheckpoint(TEST_REGION)
    checkpoint_method = getattr(s3_lightning_checkpoint, checkpoint_method_name)
    with pytest.raises(TypeError, match="argument must be a string, not"):
        checkpoint_method(**kwargs)


def test_teardown():
    s3_lightning_checkpoint = S3LightningCheckpoint(TEST_REGION)
    s3_lightning_checkpoint.teardown()
    # Assert no exception is thrown - implicit


def _test_save(
    data,
    byteorder: str,
    *,
    equal: Callable[[Any, Any], bool] = eq,
):
    s3_lightning_checkpoint = S3LightningCheckpoint(TEST_REGION)

    # Use MockClient instead of actual client.
    client = MockS3Client(TEST_REGION, TEST_BUCKET)
    s3_lightning_checkpoint._client = client

    with _patch_byteorder(byteorder):
        s3_lightning_checkpoint.save_checkpoint(data, f"s3://{TEST_BUCKET}/{TEST_KEY}")

    serialised = BytesIO(b"".join(client.get_object(TEST_BUCKET, TEST_KEY)))
    assert equal(load_with_byteorder(serialised, byteorder), data)


def _test_load(
    data,
    byteorder: str,
    *,
    equal: Callable[[Any, Any], bool] = eq,
):
    s3_lightning_checkpoint = S3LightningCheckpoint(TEST_REGION)

    # Put some data to mock bucket and use mock client
    serialised = BytesIO()
    save_with_byteorder(data, serialised, byteorder, use_modern_pytorch_format=True)
    serialised.seek(0)

    client = MockS3Client(TEST_REGION, TEST_BUCKET)
    client.add_object(TEST_KEY, serialised.read())
    s3_lightning_checkpoint._client = client

    with _patch_byteorder(byteorder):
        returned_data = s3_lightning_checkpoint.load_checkpoint(
            f"s3://{TEST_BUCKET}/{TEST_KEY}"
        )

    assert equal(returned_data, data)
