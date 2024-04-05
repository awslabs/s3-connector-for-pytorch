#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import torch
import pytest

from s3torchconnector import S3Checkpoint
from models.net import Net


@pytest.mark.parametrize(
    "tensor_dimensions",
    [[3, 2], [10, 1024, 1024]],
)
def test_general_checkpointing(checkpoint_directory, tensor_dimensions):
    tensor = torch.rand(tensor_dimensions)
    checkpoint_name = "general_checkpoint.pt"
    checkpoint = S3Checkpoint(region=checkpoint_directory.region)
    s3_uri = f"{checkpoint_directory.s3_uri}/{checkpoint_name}"
    with checkpoint.writer(s3_uri) as writer:
        torch.save(tensor, writer)

    loaded = torch.load(checkpoint.reader(s3_uri))

    assert torch.equal(tensor, loaded)


def test_nn_checkpointing(checkpoint_directory):
    nn_model = Net()
    checkpoint_name = "neural_network_model.pt"
    checkpoint = S3Checkpoint(region=checkpoint_directory.region)

    epoch = 5
    s3_uri = f"{checkpoint_directory.s3_uri}/{checkpoint_name}"
    loss = 0.4

    with checkpoint.writer(s3_uri) as writer:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": nn_model.state_dict(),
                "loss": loss,
            },
            writer,
        )

    loaded_nn_model = Net()

    # assert models are not equal before loading from checkpoint
    assert not nn_model.equals(loaded_nn_model)

    loaded_checkpoint = torch.load(checkpoint.reader(s3_uri))
    loaded_nn_model.load_state_dict(loaded_checkpoint["model_state_dict"])
    assert nn_model.equals(loaded_nn_model)

    loaded_epoch = loaded_checkpoint["epoch"]
    loaded_loss = loaded_checkpoint["loss"]
    assert loss == loaded_loss
    assert epoch == loaded_epoch

    # Assert that eval and train do not raise
    loaded_nn_model.eval()
    loaded_nn_model.train()
