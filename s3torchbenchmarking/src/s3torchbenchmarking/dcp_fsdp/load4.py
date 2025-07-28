#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import logging
import functools
from time import perf_counter, sleep
from typing import Tuple
import os
import argparse

from s3torchconnector._s3client.s3client_config import S3ClientConfig
import torch.distributed.checkpoint as dcp
import torch
import torch.distributed as dist
import torch.utils.data
from torch.distributed.elastic.multiprocessing.errors import record

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from torch.distributed.checkpoint import FileSystemReader


from s3torchconnector.dcp import S3StorageReader

from s3torchbenchmarking.models import get_benchmark_model

from s3torchbenchmarking.benchmark_utils import (
    build_checkpoint_uri,
)

Timestamps = Tuple[float, float]
logger = logging.getLogger(__name__)

import sys
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

import uuid
from datetime import datetime
from s3torchconnector.dcp import S3StorageWriter  # Add this import

def generate_random_prefix() -> str:
    """Generate a unique checkpoint prefix."""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    random_suffix = str(uuid.uuid4())[:4]
    return f"{timestamp}-{random_suffix}"

def get_writer(region: str, uri: str, suffix: str) -> S3StorageWriter:
    """Create an S3 writer for checkpointing."""
    uri = build_checkpoint_uri(uri, suffix)
    logger.info("Saving checkpoint to %s (S3) with %d copies...", uri, num_copies)
    return S3StorageWriter(region, uri, s3client_config=S3ClientConfig(
        part_size=5*1024*1024,
        throughput_target_gbps=300
    ))

def get_reader(region: str, uri: str, suffix: str) -> FileSystemReader:
    uri = build_checkpoint_uri(uri, suffix)
    logger.info("Loading checkpoint from %s (S3)...", uri)
    return S3StorageReader(region, uri, s3client_config=S3ClientConfig(
        part_size=5*1024*1024,
        throughput_target_gbps=300
    ))

@record
def run_fsdp_repeated_load(
    rank: int,
    world_size: int,
    thread_count: int,
    backend: str,
    region: str,
    uri: str,
    suffix: str,
    model_name: str = "L13b",
    checkpoint_sharding_strategy: str = "hybrid",
    num_iterations: int = 1,
    delay_between_loads: float = 0.0,
    num_copies: int = 1,
    concurrent_loads: bool = False
) -> None:
    """Execute repeated checkpoint saving and loading to stress test S3."""
    # Initialize the process group if it's not already initialized
    dist.barrier()
    
    # Create a separate process group for saving (first 8 ranks)
    save_ranks = list(range(8))
    is_save_rank = rank in save_ranks
    
    # Only create save_group if we have at least 8 ranks
    if world_size >= 8:
        # Create a process group for saving ranks
        # All ranks must call new_group, but only save_ranks will be in the group
        save_group = dist.new_group(ranks=save_ranks)
    else:
        # If fewer than 8 ranks, use the world group
        save_group = dist.group.WORLD
        is_save_rank = True  # All ranks participate in saving
    
    if rank == 0:
        logger.info(f"Starting repeated load test: {num_iterations} iterations")
        logger.info(f"Delay between loads: {delay_between_loads}s")
        logger.info(f"Concurrent loads: {concurrent_loads}")
        logger.info(f"Number of copies: {num_copies}")
    
    # Model setup
    if rank == 0:
        logger.info("Creating model")
        model_proxy = get_benchmark_model(model_name)
        model = model_proxy.model
    else:
        with torch.device("meta"):
            model_proxy = get_benchmark_model(model_name)
            model = model_proxy.model

    model_size = model_proxy.size
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

    # Create the main model for all ranks
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

    load_times = []
    save_times = []
    total_requests = 0
    failed_operations = 0

    for iteration in range(num_iterations):
        if rank == 0:
            print(f"\nIteration {iteration + 1}/{num_iterations}")
            new_suffix = f"{generate_random_prefix()}_{model_name}"
        else:
            new_suffix = None

        # Broadcast the suffix from rank 0 to all ranks
        object_list = [None]  # Initialize with None for all ranks
        if rank == 0:
            object_list[0] = new_suffix
        dist.broadcast_object_list(object_list, src=0)
        new_suffix = object_list[0]  # Get the broadcasted suffix

        # Synchronize all ranks before saving
        dist.barrier()

        if rank == 0:
            print(f"Saving checkpoint with prefix: {new_suffix}")

        # Only save ranks participate in saving
        if is_save_rank:
            start_save = perf_counter()
            storage_writer = get_writer(region, uri, new_suffix)
            
            # Use the model for saving
            with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, process_group=save_group):
                save_state_dict = {"model": model.state_dict()}
                dcp.save(save_state_dict, storage_writer=storage_writer)
                
            end_save = perf_counter()
            save_time = end_save - start_save
            
            if rank == 0:
                save_times.append(save_time)
                print(f"Save time: {save_time:.3f}s")
                
            print(f"Rank {rank} completed save operation")

        # Synchronize all ranks after saving
        dist.barrier()
        print(f"Rank {rank} reached loading phase")
        
        # All ranks participate in loading
        start_load = perf_counter()
        storage_reader = get_reader(region, uri, new_suffix)
        
        # Add a delay to ensure metadata is available
        sleep(5)
        
        print(f"Rank {rank} starting load operation")
        try:
            # All ranks load the model
            with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
                load_state_dict = {"model": model.state_dict()}
                dcp.load(load_state_dict, storage_reader=storage_reader)

            end_load = perf_counter()
            load_time = end_load - start_load
            
            if rank == 0:
                load_times.append(load_time)
                print(f"Load time: {load_time:.3f}s")
                total_requests += 1
                
            print(f"Rank {rank} completed load operation")

        except Exception as e:
            failed_operations += 1
            print(f"Rank {rank} failed to load checkpoint: {e}")
            if rank == 0:
                print(f"Failed to load checkpoint: {e}")

        # Add delay between iterations if specified
        if delay_between_loads > 0:
            sleep(delay_between_loads)

    # Final synchronization
    dist.barrier()

    # Report statistics from rank 0
    if rank == 0:
        total_save_time = sum(save_times)
        total_load_time = sum(load_times)
        avg_save_time = total_save_time / len(save_times) if save_times else 0
        avg_load_time = total_load_time / len(load_times) if load_times else 0

        print(f"\n=== Test Results ===")
        print(f"Total iterations: {num_iterations}")
        print(f"Successful operations: {len(load_times)}")
        print(f"Failed operations: {failed_operations}")
        print(f"Average save time: {avg_save_time:.3f}s")
        print(f"Average load time: {avg_load_time:.3f}s")
        print(f"Total time: {(total_save_time + total_load_time):.3f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"])
    parser.add_argument("--thread_count", type=int, default=1)
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--uri", type=str, required=True)
    
    # New arguments for repeated loading
    parser.add_argument("--iterations", type=int, default=2, 
                       help="Number of times to repeat the load operation")
    parser.add_argument("--delay", type=float, default=0.0,
                       help="Seconds to wait between load operations (0 for max throughput)")
    parser.add_argument("--concurrent", action="store_true",
                       help="Don't synchronize between ranks for maximum concurrency")
    parser.add_argument("--model", type=str, default="L7b",
                       help="Model name to use for benchmarking")
    parser.add_argument("--num_copies", type=int, default=1,
                       help="Number of checkpoint copies to create")
    
    args = parser.parse_args()
    
    print("@Started backend")
    backend = args.backend
    
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ['LOCAL_RANK'])
    import socket 
    print(f"Starting on host {socket.gethostname()} with global rank {rank}, local_rank {local_rank}, world_size {world_size}")
    
    thread_count = args.thread_count
    if not dist.is_initialized():
        dist.init_process_group(args.backend, rank=rank, world_size=world_size)

    region = args.region
    uri = args.uri
    suffix = ""
    checkpoint_sharding_strategy = "hybrid"
    num_copies = args.num_copies

    run_fsdp_repeated_load(
        rank,
        world_size,
        thread_count,
        backend,
        region,
        uri,
        suffix,
        model_name=args.model,
        checkpoint_sharding_strategy=checkpoint_sharding_strategy,
        num_iterations=args.iterations,
        num_copies=num_copies,
        delay_between_loads=args.delay,
        concurrent_loads=args.concurrent
    ) 
    
    if dist.is_initialized():
        dist.destroy_process_group()

