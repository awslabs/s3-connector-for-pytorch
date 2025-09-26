#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import logging
import functools
from time import perf_counter
from typing import Tuple
import os
import socket
import argparse
import uuid
from datetime import datetime

import torch.distributed.checkpoint as dcp
import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from s3torchbenchmarking.models import get_benchmark_model
from s3torchbenchmarking.benchmark_utils import (
    build_checkpoint_uri,
)
from s3torchconnector._s3client.s3client_config import S3ClientConfig
from s3torchconnector.dcp import S3StorageWriter  # Add this import


Timestamps = Tuple[float, float]
logger = logging.getLogger(__name__)


def generate_random_prefix() -> str:
    """Generate a unique checkpoint prefix."""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    random_suffix = str(uuid.uuid4())[:4]
    return f"{timestamp}-{random_suffix}"


def get_writer(region: str, uri: str, suffix: str) -> S3StorageWriter:
    """Create an S3 writer for checkpointing."""
    uri = build_checkpoint_uri(uri, suffix)
    logger.info("Saving checkpoint to %s (S3)...", uri)
    return S3StorageWriter(
        region,
        uri,
        s3client_config=S3ClientConfig(
            part_size=5 * 1024 * 1024, throughput_target_gbps=300
        ),
        num_copies=20,
    )


@record
def run_fsdp_save(
    rank: int,
    backend: str,
    region: str,
    uri: str,
    model_name: str = "L7b",
    checkpoint_sharding_strategy: str = "hybrid",
    num_iterations: int = 2,
) -> None:
    """Execute repeated checkpoint saving and loading to stress test S3."""
    # [Previous initialization code remains the same until after FSDP wrapping]
    # Initialize the process group if it's not already initialized
    dist.barrier()

    if rank == 0:
        logger.info(f"Starting repeated Save test: {num_iterations} iterations")

    # Model setup (same as original)
    if rank == 0:
        logger.info("Creating model")
        model_proxy = get_benchmark_model(model_name)
        model = model_proxy.model
    else:
        with torch.device("meta"):
            model_proxy = get_benchmark_model(model_name)
            model = model_proxy.model

    model_name = model_proxy.name
    if rank == 0:
        logger.info(f"Model {model_name} created")

    transformer_layer = LlamaDecoderLayer
    gpt_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            transformer_layer,
        },
    )

    if backend == "nccl":
        device_id = rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        param_init_fn = lambda module: module.to_empty(
            device=torch.device("cuda"), recurse=False
        )
    else:
        device_id = rank % torch.cpu.device_count()
        torch.cpu.set_device(device_id)
        param_init_fn = lambda module: module.to_empty(
            device=torch.device("cpu"), recurse=False
        )

    if checkpoint_sharding_strategy == "full":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif checkpoint_sharding_strategy == "hybrid":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError("Available sharding strategies are full and hybrid")

    model = FSDP(
        model,
        auto_wrap_policy=gpt_auto_wrap_policy,
        device_id=(
            torch.cuda.current_device()
            if backend == "nccl"
            else torch.cpu.current_device()
        ),
        use_orig_params=False,
        sharding_strategy=sharding_strategy,
        sync_module_states=True if backend == "nccl" else False,
        param_init_fn=param_init_fn if rank != 0 else None,
    )

    if rank == 0:
        print("Wrapped model with FSDP")

    # Prepare state dict structure
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        state_dict = {
            "model": model.state_dict(),
        }

    save_times = []

    for iteration in range(num_iterations):
        if rank == 0:
            print(f"\nIteration {iteration + 1}/{num_iterations}")
            new_suffix = generate_random_prefix()
        else:
            new_suffix = None

        # Create a list to hold the object for broadcasting

        object_list = [None]  # Initialize with None for all ranks
        if rank == 0:
            object_list[0] = new_suffix

        # Broadcast the suffix from rank 0 to all ranks
        dist.broadcast_object_list(object_list, src=0)
        new_suffix = object_list[0]  # Get the broadcasted suffix

        if rank == 0:
            print(f"Saving checkpoint with prefix: {new_suffix}")

        start_save = perf_counter()
        storage_writer = get_writer(region, uri, new_suffix)
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            dcp.save(state_dict, storage_writer=storage_writer)
        end_save = perf_counter()
        save_time = end_save - start_save
        save_times.append(save_time)

        if rank == 0:
            print(f"Save time: {save_time:.3f}s")

        # Load the checkpoint we just saved using multiple workers
        dist.barrier()

    if rank == 0:
        total_save_time = sum(save_times)
        avg_save_time = total_save_time / len(save_times) if save_times else 0

        print(f"\n=== Test Results ===")
        print(f"Total iterations: {num_iterations}")
        print(f"Average save time: {avg_save_time:.3f}s")

        # Calculate approximate S3 request rates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # [Previous arguments remain the same]
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"])
    parser.add_argument("--thread_count", type=int, default=1)
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--uri", type=str, required=True)

    # New arguments for repeated loading
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of times to repeat the load operation",
    )
    parser.add_argument(
        "--model", type=str, default="L7b", help="Model name to use for benchmarking"
    )

    args = parser.parse_args()

    backend = args.backend

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    print(
        f"Starting on host {socket.gethostname()} with global rank {rank}, local_rank {local_rank}, world_size {world_size}"
    )

    thread_count = args.thread_count
    if not dist.is_initialized():
        dist.init_process_group(args.backend, rank=rank, world_size=world_size)

    region = args.region
    uri = args.uri
    # Add new argument for number of workers
    checkpoint_sharding_strategy = "hybrid"

    args = parser.parse_args()

    # [Rest of the main block remains the same, but add workers parameter to function call]
    run_fsdp_save(
        rank=rank,
        backend=backend,
        region=region,
        uri=uri,
        checkpoint_sharding_strategy=checkpoint_sharding_strategy,
        num_iterations=args.iterations,
    )
    if dist.is_initialized():
        dist.destroy_process_group()
