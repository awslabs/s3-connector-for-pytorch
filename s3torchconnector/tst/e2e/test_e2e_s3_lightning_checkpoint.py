#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
import pytest
import random
import torch

from pathlib import Path
from typing import Dict, Any
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos import WikiText2
from lightning.pytorch.plugins import AsyncCheckpointIO
from torch.utils.data import DataLoader

from s3torchconnector import S3Checkpoint
from s3torchconnector._s3client import S3Client
from s3torchconnector._s3dataset_common import parse_s3_uri
from s3torchconnector.lightning import S3LightningCheckpoint
from s3torchconnectorclient import S3Exception

from models.net import Net
from models.lightning_transformer import LightningTransformer, L


def test_save_and_load_checkpoint(checkpoint_directory):
    tensor = torch.rand(3, 10, 10)
    s3_lightning_checkpoint = S3LightningCheckpoint(region=checkpoint_directory.region)
    checkpoint_name = "lightning_checkpoint.ckpt"
    s3_uri = f"{checkpoint_directory.s3_uri}{checkpoint_name}"
    s3_lightning_checkpoint.save_checkpoint(tensor, s3_uri)
    loaded_checkpoint = s3_lightning_checkpoint.load_checkpoint(s3_uri)
    assert torch.equal(tensor, loaded_checkpoint)


def test_load_compatibility_with_s3_checkpoint(checkpoint_directory):
    tensor = torch.rand(3, 10, 10)
    checkpoint_name = "general_checkpoint.pt"
    checkpoint = S3Checkpoint(region=checkpoint_directory.region)
    s3_uri = f"{checkpoint_directory.s3_uri}{checkpoint_name}"
    with checkpoint.writer(s3_uri) as writer:
        torch.save(tensor, writer)
    lightning_checkpoint = S3LightningCheckpoint(region=checkpoint_directory.region)
    loaded_checkpoint = lightning_checkpoint.load_checkpoint(s3_uri)
    assert torch.equal(tensor, loaded_checkpoint)


def test_save_compatibility_with_s3_checkpoint(checkpoint_directory):
    tensor = torch.rand(3, 10, 10)
    checkpoint_name = "lightning_checkpoint.ckpt"
    lightning_checkpoint = S3LightningCheckpoint(region=checkpoint_directory.region)
    s3_uri = f"{checkpoint_directory.s3_uri}{checkpoint_name}"
    lightning_checkpoint.save_checkpoint(tensor, s3_uri)
    checkpoint = S3Checkpoint(region=checkpoint_directory.region)
    loaded_checkpoint = torch.load(checkpoint.reader(s3_uri))
    assert torch.equal(tensor, loaded_checkpoint)


def test_delete_checkpoint(checkpoint_directory):
    tensor = torch.rand(3, 10, 10)
    checkpoint_name = "lightning_checkpoint.ckpt"
    lightning_checkpoint = S3LightningCheckpoint(region=checkpoint_directory.region)
    s3_uri = f"{checkpoint_directory.s3_uri}{checkpoint_name}"
    lightning_checkpoint.save_checkpoint(tensor, s3_uri)
    loaded_checkpoint = lightning_checkpoint.load_checkpoint(s3_uri)
    assert torch.equal(tensor, loaded_checkpoint)
    lightning_checkpoint.remove_checkpoint(s3_uri)
    with pytest.raises(S3Exception, match="Service error: The key does not exist"):
        lightning_checkpoint.load_checkpoint(s3_uri)


def test_load_trained_checkpoint(checkpoint_directory):
    nonce = random.randrange(2**64)
    dataset = WikiText2(data_dir=Path(f"/tmp/data/{nonce}"))
    dataloader = DataLoader(dataset, num_workers=3)
    model = LightningTransformer(vocab_size=dataset.vocab_size)
    trainer = L.Trainer(fast_dev_run=2)
    trainer.fit(model=model, train_dataloaders=dataloader)
    checkpoint_name = "lightning_module_training_checkpoint.pt"
    s3_uri = f"{checkpoint_directory.s3_uri}{checkpoint_name}"
    trainer.save_checkpoint(s3_uri)
    s3_lightning_checkpoint = S3LightningCheckpoint(region=checkpoint_directory.region)
    loaded_checkpoint = s3_lightning_checkpoint.load_checkpoint(s3_uri)
    _verify_equal_state_dict(model.state_dict(), loaded_checkpoint["state_dict"])


def test_compatibility_with_trainer_plugins(checkpoint_directory):
    nonce = random.randrange(2**64)
    dataset = WikiText2(data_dir=Path(f"/tmp/data/{nonce}"))
    dataloader = DataLoader(dataset, num_workers=3)
    model = LightningTransformer(vocab_size=dataset.vocab_size)
    s3_lightning_checkpoint = S3LightningCheckpoint(region=checkpoint_directory.region)
    trainer = L.Trainer(
        default_root_dir=checkpoint_directory.s3_uri,
        plugins=[s3_lightning_checkpoint],
        max_epochs=1,
        max_steps=3,
    )
    trainer.fit(model, dataloader)
    checkpoint_key = "lightning_logs/version_0/checkpoints/epoch=0-step=3.ckpt"
    checkpoint_s3_uri = f"{checkpoint_directory.s3_uri}{checkpoint_key}"
    loaded_checkpoint = s3_lightning_checkpoint.load_checkpoint(checkpoint_s3_uri)
    _verify_equal_state_dict(model.state_dict(), loaded_checkpoint["state_dict"])

    new_model = LightningTransformer.load_from_checkpoint(
        checkpoint_s3_uri, vocab_size=dataset.vocab_size
    )
    _verify_equal_state_dict(model.state_dict(), new_model.state_dict())


def test_compatibility_with_checkpoint_callback(checkpoint_directory):
    nonce = random.randrange(2**64)
    dataset = WikiText2(data_dir=Path(f"/tmp/data/{nonce}"))
    dataloader = DataLoader(dataset, num_workers=3)

    model = LightningTransformer(vocab_size=dataset.vocab_size)
    s3_lightning_checkpoint = S3LightningCheckpoint(checkpoint_directory.region)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_directory.s3_uri,
        save_top_k=1,
        every_n_epochs=1,
        filename="checkpoint-{epoch:02d}-{step:02d}",
        enable_version_counter=True,
    )
    trainer = L.Trainer(
        plugins=[s3_lightning_checkpoint],
        callbacks=[checkpoint_callback],
        min_epochs=4,
        max_epochs=5,
        max_steps=3,
    )

    trainer.fit(model, dataloader)
    expected_checkpoint_name = "checkpoint-epoch=00-step=03.ckpt"
    bucket, prefix = parse_s3_uri(checkpoint_directory.s3_uri)
    s3_client = S3Client(region=checkpoint_directory.region)
    list_result = list(s3_client.list_objects(bucket, prefix))
    assert list_result is not None
    assert len(list_result) == 1
    assert str.endswith(list_result[0].object_info[0].key, expected_checkpoint_name)

    checkpoint_s3_uri = f"{checkpoint_directory.s3_uri}{expected_checkpoint_name}"
    loaded_checkpoint = s3_lightning_checkpoint.load_checkpoint(checkpoint_s3_uri)
    _verify_equal_state_dict(model.state_dict(), loaded_checkpoint["state_dict"])


def test_compatibility_with_async_checkpoint_io(checkpoint_directory):
    nonce = random.randrange(2**64)
    dataset = WikiText2(data_dir=Path(f"/tmp/data/{nonce}"))
    dataloader = DataLoader(dataset, num_workers=3)

    model = LightningTransformer(vocab_size=dataset.vocab_size)
    s3_lightning_checkpoint = S3LightningCheckpoint(checkpoint_directory.region)
    async_s3_lightning_checkpoint = AsyncCheckpointIO(s3_lightning_checkpoint)

    trainer = L.Trainer(
        default_root_dir=checkpoint_directory.s3_uri,
        plugins=[async_s3_lightning_checkpoint],
        min_epochs=4,
        max_epochs=5,
        max_steps=3,
    )

    trainer.fit(model, dataloader)

    # Ensure that all the running futures have finished executing
    async_s3_lightning_checkpoint.teardown()

    checkpoint_key = "lightning_logs/version_0/checkpoints/epoch=0-step=3.ckpt"
    checkpoint_s3_uri = f"{checkpoint_directory.s3_uri}{checkpoint_key}"
    loaded_checkpoint = s3_lightning_checkpoint.load_checkpoint(checkpoint_s3_uri)
    _verify_equal_state_dict(model.state_dict(), loaded_checkpoint["state_dict"])


def test_nn_checkpointing(checkpoint_directory):
    nn_model = Net()
    checkpoint_name = "lightning_neural_network_model.pt"
    s3_lightning_checkpoint = S3LightningCheckpoint(region=checkpoint_directory.region)
    s3_uri = f"{checkpoint_directory.s3_uri}{checkpoint_name}"
    epoch = 5
    loss = 0.4

    s3_lightning_checkpoint.save_checkpoint(
        {
            "epoch": epoch,
            "model_state_dict": nn_model.state_dict(),
            "loss": loss,
        },
        s3_uri,
    )

    loaded_nn_model = Net()

    # Assert models are not equal before loading from checkpoint
    assert not nn_model.equals(loaded_nn_model)

    loaded_checkpoint = s3_lightning_checkpoint.load_checkpoint(s3_uri)
    loaded_nn_model.load_state_dict(loaded_checkpoint["model_state_dict"])
    assert nn_model.equals(loaded_nn_model)

    loaded_epoch = loaded_checkpoint["epoch"]
    loaded_loss = loaded_checkpoint["loss"]
    assert loss == loaded_loss
    assert epoch == loaded_epoch

    # Assert that eval and train do not raise
    loaded_nn_model.eval()
    loaded_nn_model.train()


def _verify_equal_state_dict(
    state_dict: Dict[str, Any], loaded_state_dict: Dict[str, Any]
):
    for (model_key, model_value), (loaded_key, loaded_value) in zip(
        state_dict.items(), loaded_state_dict.items()
    ):
        # These are tuples (str, Tensor)
        assert model_key == loaded_key
        assert torch.equal(model_value, loaded_value)
