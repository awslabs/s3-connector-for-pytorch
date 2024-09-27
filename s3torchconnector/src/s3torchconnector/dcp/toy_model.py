import torch
import torch.distributed.checkpoint as DCP
from torch import nn
import argparse
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from s3torchconnector import S3StorageWriter, S3StorageReader, S3DPWriter, S3DPReader

CHECKPOINT_DIR = "checkpoint"

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(16, 16)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(16, 8)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def setup(backend):
    dist.init_process_group(backend)


def cleanup():
    dist.destroy_process_group()


def run_fsdp_checkpoint_save_example(rank, backend):
    print(f"Running basic FSDP checkpoint saving example on rank {rank}.")

    # Need to put tensor on a GPU device for nccl backend
    if backend == "nccl":
        device_id = rank % torch.cuda.device_count()
        model = ToyModel().to(device_id)
        model = FSDP(model, device_id=device_id)
    elif backend == "gloo":
        model = ToyModel().to(device=torch.device("cpu"))
        model = FSDP(model, device_id=torch.cpu.current_device())
    else:
        raise Exception(f"Unknown backend type: {backend}")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    optimizer.zero_grad()
    if backend == "nccl":
        model(torch.rand(8, 16, device=torch.device("cuda"))).sum().backward()
    else:
        model(torch.rand(8, 16, device=torch.device("cpu"))).sum().backward()
    optimizer.step()

    # Set FSDP StateDictType to SHARDED_STATE_DICT to checkpoint the model state dict
    FSDP.set_state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
    )
    state_dict = {
        "model": model.state_dict(),
    }

    thread_count = 4
    bucket = "dcp-poc-test-3"
    path = f"s3://{bucket}/epoch_1/"
    region = "eu-west-2"
    writer_to_use = "s3_fs"
    writer = get_writer(region, path, thread_count, writer_to_use)

    DCP.save(state_dict=state_dict, storage_writer=writer)

    print("Checkpoint saved for epoch 1.")

    # Save for another epoch
    state_dict = {
        "model": model.state_dict(),
        "prefix": "bla",
    }
    optimizer.step()

    path = f"s3://{bucket}/epoch_2/"
    writer = get_writer(region, path, thread_count, writer_to_use)
    DCP.save(state_dict=state_dict, storage_writer=writer)

    print("Checkpoint saved for epoch 2.")

def run_fsdp_checkpoint_load_example(rank, backend):
    print(f"Running basic FSDP checkpoint loading example on rank {rank}.")

    # Need to put tensor on a GPU device for nccl backend
    if backend == "nccl":
        device_id = rank % torch.cuda.device_count()
        model = ToyModel().to(device_id)
        model = FSDP(model, device_id=device_id)
    elif backend == "gloo":
        model = ToyModel().to(device=torch.device("cpu"))
        model = FSDP(model, device_id=torch.cpu.current_device())
    else:
        raise Exception(f"Unknown backend type: {backend}")

    # Set FSDP StateDictType to SHARDED_STATE_DICT to load the sharded model state dict
    FSDP.set_state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
    )

    # Prepare state_dict to load into
    loaded_state_dict = {"model": model.state_dict(),}

    thread_count = 1
    bucket = "dcp-poc-test-3"
    path = f"s3://{bucket}/epoch_2/"
    region = "eu-west-2"
    reader_to_use = "s3_fs"
    reader = get_reader(region, path, thread_count, reader_to_use)

    # Load the checkpoint
    DCP.load(state_dict=loaded_state_dict, storage_reader=reader)

    # Load the model state dict from the checkpoint
    model.load_state_dict(loaded_state_dict["model"])
    print("Checkpoint loaded and model state dict restored.")

    print(loaded_state_dict)


def get_writer(region, path, thread_count, writer_to_use):
    if writer_to_use == "local":
        writer = DCP.FileSystemWriter(CHECKPOINT_DIR, single_file_per_rank=True)
    elif writer_to_use == "s3_fs":
        writer = S3DPWriter(region=region, path=path, thread_count=thread_count)
    else:
        writer = S3StorageWriter(region=region, s3_uri=path, thread_count=thread_count)
    return writer

def get_reader(region, path, thread_count, reader_to_use):
    if reader_to_use == "local":
        reader = DCP.FileSystemReader(CHECKPOINT_DIR)
    elif reader_to_use == "s3_fs":
        reader = S3DPReader(region=region, path=path)
    else:
        reader = S3StorageReader(region=region, s3_uri=path, thread_count=thread_count)
    return reader

if __name__ == "__main__":
    """
    How to use:
    Step 1: Set up EC2 Instances
        Create two EC2 instances on AWS.
        Modify the security groups of these instances to allow inbound TCP/UDP connections on all ports between them.

    Step 2: Designate Master and Worker Hosts. Choose one instance as the master host. In this example, we'll assume
    the master host's IP address is 172.31.18.217. Decide on the port number the master host will listen on
    for the worker host. For this guide, we'll use port 1234.

    Step 3: Run Commands on Master and Worker Hosts. For CPU training, run the following commands simultaneously
    on the master and worker hosts:
    Master Host:
        torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=172.31.18.217 --master_port=1234 toy_model.py --backend=gloo
    Worker Host:
        torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=172.31.18.217 --master_port=1234 toy_model.py --backend=gloo

    Here's what each argument means:
        --nproc_per_node=4: Run four processes on each node (instance).
        --nnodes=2: Use two nodes (instances) for training.
        --node_rank=0 (master) / --node_rank=1 (worker): Set the rank of the current node.
        --master_addr=172.31.18.217: Set the IP address of the master host. Use private IP address.
        --master_port=1234: Set the port number the master host is listening on.
        toy_model.py: The script to run for training.
        --backend=gloo: Use the gloo backend, which utilizes CPU for training.

    For GPU training, run the following commands simultaneously on the master and worker hosts:
    Master Host:
        torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=172.31.18.217 --master_port=1234 toy_model.py --backend=nccl
    Worker Host:
        torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=172.31.18.217 --master_port=1234 toy_model.py --backend=nccl

    The meaningful difference from the CPU training commands is the --backend=nccl argument, which uses the nccl
    backend for GPU training. Also, nproc_per_node was set to 1 to support running that command on instance with only
    one GPU. When you use GPU for training,you need to limit amount of processes per node (nproc_per_node),
    by amount of GPU available on node.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"])
    parser.add_argument("--action", type=str, default="save", choices=["save", "load"], help="Action to perform: 'save' or 'load'.")
    args = parser.parse_args()

    setup(args.backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Starting for rank {rank}, world_size is {world_size}")

    if args.action == "save":
        run_fsdp_checkpoint_save_example(rank, args.backend)
    elif args.action == "load":
        run_fsdp_checkpoint_load_example(rank, args.backend)

    cleanup()