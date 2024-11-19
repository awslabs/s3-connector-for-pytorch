#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

# inspired by https://github.com/pytorch/tutorials/blob/main/recipes_source/distributed_checkpoint_recipe.rst#loading


import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
import torch.multiprocessing as mp
import torch.nn as nn

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from s3torchconnector.dcp import S3StorageReader


class AppState(Stateful):
    """This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.

    Note: We take advantage of this wrapper to hande calling distributed state dict methods on the model
    and optimizer.
    """

    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(
            self.model, self.optimizer
        )
        return {"model": model_state_dict, "optim": optimizer_state_dict}

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that we've loaded
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(16, 16)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(16, 8)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355 "

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def run_fsdp_checkpoint_load_example(rank, world_size, region, s3_uri):
    print(f"Running basic FSDP checkpoint loading example on rank {rank}.")
    setup(rank, world_size)

    # create a model and move it to GPU with id rank
    model = ToyModel().to(rank)
    model = FSDP(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    state_dict = {"app": AppState(model, optimizer)}
    storage_reader = S3StorageReader(region, s3_uri)
    dcp.load(state_dict=state_dict, storage_reader=storage_reader)

    print(f"Model loaded on rank:{rank}")
    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    region = os.getenv("REGION")
    s3_uri = os.getenv("CHECKPOINT_PATH")
    print(f"Running fsdp load checkpoint example on {world_size} devices.")
    mp.spawn(
        run_fsdp_checkpoint_load_example,
        args=(world_size, region, s3_uri),
        nprocs=world_size,
        join=True,
    )
