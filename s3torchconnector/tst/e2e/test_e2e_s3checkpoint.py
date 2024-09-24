#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import torch
import pytest

from s3torchconnector import S3Checkpoint
from models.net import Net
from concurrent.futures import ThreadPoolExecutor, as_completed

@pytest.mark.parametrize(
    "tensor_dimensions",
    [[3, 2], [10, 1024, 1024]],
)
def test_general_checkpointing(checkpoint_directory, tensor_dimensions):
    tensor = torch.rand(tensor_dimensions)
    checkpoint_name = "general_checkpoint.pt"
    checkpoint = S3Checkpoint(region=checkpoint_directory.region)
    s3_uri = f"{checkpoint_directory.s3_uri}/{checkpoint_name}"
    save_checkpoint(s3_uri, checkpoint, tensor)

    loaded = torch.load(checkpoint.reader(s3_uri))

    assert torch.equal(tensor, loaded)


def test_nn_checkpointing(checkpoint_directory):
    nn_model = Net()
    checkpoint_name = "neural_network_model.pt"
    checkpoint = S3Checkpoint(region=checkpoint_directory.region)

    epoch = 5
    s3_uri = f"{checkpoint_directory.s3_uri}/{checkpoint_name}"
    loss = 0.4

    save_checkpoint(s3_uri, checkpoint, {
                "epoch": epoch,
                "model_state_dict": nn_model.state_dict(),
                "loss": loss,
            })

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

def test_parallel_checkpointing(checkpoint_directory):
    checkpoint = S3Checkpoint(region=checkpoint_directory.region)
    tensor = torch.rand(2,3)
    results = []
    futures = []

    with ThreadPoolExecutor(max_workers=100) as executor:
        for i in range(100):
            checkpoint_name = f"general_checkpoint.pt_{i}"
            s3_uri = f"{checkpoint_directory.s3_uri}/{checkpoint_name}"
            futures.append(executor.submit(save_checkpoint, s3_uri, checkpoint, tensor))

        for future in as_completed(futures):
            try:
                result = future.result()  # Get the result of the completed future
                results.append(result)
            except AssertionError as e:
                results.append(f"AssertionError: {e}")
            except Exception as e:
                results.append(f"Error: {e}")

    assert not any("AssertionError" in str(res) for res in results), "An AssertionError was encountered."

def save_checkpoint(s3_uri, checkpoint: S3Checkpoint, checkpoint_content):
    with checkpoint.writer(s3_uri) as writer:
        torch.save(checkpoint_content, writer)


