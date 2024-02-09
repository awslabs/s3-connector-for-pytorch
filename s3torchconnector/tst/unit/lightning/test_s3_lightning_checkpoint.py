#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from io import BytesIO
from operator import eq
from pathlib import Path
from typing import Callable, Any

import hypothesis
import pytest
import torch
from hypothesis import given, HealthCheck
from lightning.fabric.plugins import CheckpointIO
from lightning.pytorch.plugins import AsyncCheckpointIO

from s3torchconnector._s3client import MockS3Client
from s3torchconnector.lightning import S3LightningCheckpoint
from .._checkpoint_byteorder_patch import (
    byteorders,
    save_with_byteorder,
    load_with_byteorder,
    _patch_byteorder,
)
from .._hypothesis_python_primitives import python_primitives
from s3torchconnectorclient import S3Exception

TEST_BUCKET = "test-bucket"
TEST_KEY = "test-key"
TEST_REGION = "us-east-1"


@pytest.fixture()
def client() -> MockS3Client:
    return MockS3Client(TEST_REGION, TEST_BUCKET)


@pytest.fixture()
def lightning_checkpoint(client) -> S3LightningCheckpoint:
    s3_lightning_checkpoint = S3LightningCheckpoint(TEST_REGION)
    s3_lightning_checkpoint._client = client
    return s3_lightning_checkpoint


@pytest.fixture()
def async_lightning_checkpoint(lightning_checkpoint) -> Callable[[], AsyncCheckpointIO]:
    return lambda: AsyncCheckpointIO(lightning_checkpoint)


@hypothesis.settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(python_primitives, byteorders)
def test_lightning_checkpointing_saves_python_primitives(
    client, lightning_checkpoint, data, byteorder
):
    _test_save(client, lightning_checkpoint, data, byteorder)


@hypothesis.settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(byteorders)
def test_lightning_checkpointing_saves_tensor(client, lightning_checkpoint, byteorder):
    tensor = torch.rand(2, 4)
    _test_save(client, lightning_checkpoint, tensor, byteorder, equal=torch.equal)


@hypothesis.settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(byteorders)
def test_async_lightning_checkpointing_saves_tensor(
    client, async_lightning_checkpoint, byteorder
):
    tensor = torch.rand(2, 4)
    _test_save(
        client, async_lightning_checkpoint(), tensor, byteorder, equal=torch.equal
    )


@hypothesis.settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(byteorders)
def test_lightning_checkpointing_saves_untyped_storage(
    client, lightning_checkpoint, byteorder
):
    storage = torch.UntypedStorage([1, 2, 3])
    _test_save(
        client,
        lightning_checkpoint,
        storage,
        byteorder,
        equal=lambda a, b: list(a) == list(b),
    )


@hypothesis.settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(python_primitives, byteorders)
def test_lightning_checkpointing_loads_python_primitives(
    client, lightning_checkpoint, data, byteorder
):
    _test_load(client, lightning_checkpoint, data, byteorder)


@hypothesis.settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(byteorders)
def test_lightning_checkpointing_loads_tensor(client, lightning_checkpoint, byteorder):
    tensor = torch.rand(2, 4)
    _test_load(client, lightning_checkpoint, tensor, byteorder, equal=torch.equal)


@hypothesis.settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(byteorders)
def test_lightning_checkpointing_loads_untyped_storage(
    client, lightning_checkpoint, byteorder
):
    storage = torch.UntypedStorage([1, 2, 3])
    _test_load(
        client,
        lightning_checkpoint,
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
        ("save_checkpoint", {"path": Path("bucket", "key"), "checkpoint": None}),
        ("load_checkpoint", {"path": Path("/", "bucket", "key")}),
        ("remove_checkpoint", {"path": Path()}),
        ("remove_checkpoint", {"path": ["not", "a", "string"]}),
    ],
)
def test_invalid_path(lightning_checkpoint, checkpoint_method_name, kwargs):
    checkpoint_method = getattr(lightning_checkpoint, checkpoint_method_name)
    with pytest.raises(TypeError, match="argument must be a string, not"):
        checkpoint_method(**kwargs)


def test_teardown(lightning_checkpoint):
    lightning_checkpoint.teardown()
    # Assert no exception is thrown - implicit


def _test_save(
    client,
    checkpoint: CheckpointIO,
    data,
    byteorder: str,
    *,
    equal: Callable[[Any, Any], bool] = eq,
):
    with _patch_byteorder(byteorder):
        checkpoint.save_checkpoint(data, f"s3://{TEST_BUCKET}/{TEST_KEY}")
        # For async checkpointing, ensure that we finish writing the checkpoint before we un-patch the byteorder
        checkpoint.teardown()

    serialised = BytesIO(b"".join(client.get_object(TEST_BUCKET, TEST_KEY)))
    assert equal(load_with_byteorder(serialised, byteorder), data)


def _test_load(
    client,
    checkpoint: CheckpointIO,
    data,
    byteorder: str,
    *,
    equal: Callable[[Any, Any], bool] = eq,
):
    # Put some data to mock bucket and use mock client
    serialised = BytesIO()
    save_with_byteorder(data, serialised, byteorder, use_modern_pytorch_format=True)
    serialised.seek(0)
    client.add_object(TEST_KEY, serialised.read())

    with _patch_byteorder(byteorder):
        returned_data = checkpoint.load_checkpoint(f"s3://{TEST_BUCKET}/{TEST_KEY}")

    assert equal(returned_data, data)
