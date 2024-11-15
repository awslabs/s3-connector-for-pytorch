# based on PyTorch sample https://github.com/pytorch/pytorch/blob/main/torch/distributed/checkpoint/examples/stateful_example.py

import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    _patch_model_state_dict,
    _patch_optimizer_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from s3torchconnector.dcp import S3StorageWriter, S3StorageReader


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.net1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.net3 = nn.Linear(32, 64)
        self.net4 = nn.Sequential(nn.ReLU(), nn.Linear(64, 8))

    def forward(self, x):
        return self.net4(self.net3(self.net2(self.net1(x))))

    def get_input(self):
        return torch.rand(8, 8, device="cuda")


def _make_stateful(model, optim):
    _patch_model_state_dict(model)
    _patch_optimizer_state_dict(model, optimizers=optim)


def _train(model, optim, train_steps=1):
    torch.manual_seed(0)
    loss = None
    for _ in range(train_steps):
        loss = model(model.get_input()).sum()
        loss.backward()
        optim.step()
        optim.zero_grad()

    return loss


def _init_model(device, world_size):
    device_mesh = init_device_mesh(device, (world_size,))
    model = Model().cuda()
    model = FSDP(
        model,
        device_mesh=device_mesh,
        use_orig_params=True,
    )
    optim = torch.optim.Adam(model.parameters(), lr=0.1)
    _make_stateful(model, optim)

    return model, optim

def _compare_models(model1, model2, rank, rtol=1e-5, atol=1e-8):
    model1.eval()
    model2.eval()

    with FSDP.summon_full_params(model1), FSDP.summon_full_params(model2):
        for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
            if name1 != name2:
                print(f"Parameter names don't match: {name1} vs {name2}. Rank:{rank}")
                return False

            if not torch.allclose(param1, param2, rtol=rtol, atol=atol):
                print(f"Parameters don't match for {name1}. Rank:{rank}")
                print(f"Max difference: {(param1 - param2).abs().max().item()}. Rank:{rank}")
                return False

    print("All parameters match within the specified tolerance. Rank:{rank}")
    return True

def run(rank, world_size, device="cuda"):
    # Set up world pg
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    print(f"Train initial model on rank:{rank}")
    model, optim = _init_model(device, world_size)
    _train(model, optim, train_steps=2)

    region = "us-east-1"
    s3_uri = f"s3://PLEASE-PROVIDE-ACTUAL-BUCKET-NAME/"

    print(f"Saving checkpoint on rank:{rank}")
    # initialize S3StorageWriter with region and bucket name, before passing to dcp.save as writer
    storage_writer = S3StorageWriter(region, s3_uri)
    dcp.save(
        state_dict={"model": model, "optimizer": optim},
        storage_writer=storage_writer,
    )

    # presumably do something else
    print(f"Train another model on rank:{rank}")
    loaded_model, loaded_optim = _init_model(device, world_size)
    _train(loaded_model, loaded_optim, train_steps=4)

    print(f"Check that models are different on rank:{rank}")
    assert _compare_models(model, loaded_model, rank) is False

    print(f"Load checkpoint on rank:{rank}")
    #  initialize S3StorageReader with region and bucket name, before passing to dcp.load as reader
    storage_reader = S3StorageReader(region, s3_uri)
    dcp.load(
        state_dict={"model": loaded_model, "optimizer": loaded_optim},
        storage_reader=storage_reader
    )
    print(f"Check that loaded model and original model are the same on rank:{rank}")
    assert _compare_models(model, loaded_model, rank)

    print(f"Train loaded model on rank:{rank}")
    _train(loaded_model, loaded_optim, train_steps=2)

    print(f"Quiting on rank:{rank}")

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Running stateful checkpoint example on {world_size} devices.")
    mp.spawn(
        run,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )