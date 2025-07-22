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

def get_reader(region:str, uri: str, suffix: str) -> FileSystemReader:
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
    model_name: str = "L7b",
    checkpoint_sharding_strategy: str = "full",
    num_iterations: int = 5,
    delay_between_loads: float = 0.0,
    concurrent_loads: bool = False
) -> None:
    """Execute repeated checkpoint loading to stress test S3.
    
    Args:
        num_iterations: Number of times to repeat the load operation
        delay_between_loads: Seconds to wait between load operations (0 for maximum throughput)
        concurrent_loads: If True, don't synchronize between ranks for maximum concurrency
    """
    # Initialize the process group if it's not already initialized
    dist.barrier()
    
    if rank == 0:
        logger.info(f"Starting repeated load test: {num_iterations} iterations")
        logger.info(f"Delay between loads: {delay_between_loads}s")
        logger.info(f"Concurrent loads: {concurrent_loads}")
    
    # Model setup (same as original)
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
        state_dict_template = {
            "model": model.state_dict(),
        }

    # Storage reader setup
    storage_reader = get_reader(region, uri, suffix)
    
    # Statistics tracking
    load_times = []
    total_requests = 0
    failed_loads = 0
    
    if rank == 0:
        print(f"Starting {num_iterations} repeated loads...")
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        # Create state dict once and reuse
        state_dict = {
            "model": model.state_dict(),
        }
    
        for iteration in range(num_iterations):
        
            if not concurrent_loads:
                dist.barrier()

            start_load = perf_counter()
            dcp.load(state_dict, storage_reader=storage_reader)
            end_load = perf_counter()
            # Perform the load operation
            dcp.load(state_dict, storage_reader=storage_reader)
            
            # Verify loading worked
            end_load = perf_counter()
            load_time = end_load - start_load
            load_times.append(load_time)
            total_requests += 1
            
            if rank == 0:
                print(f"Iteration {iteration + 1}/{num_iterations}: "
                      f"Load time: {load_time:.3f}s, ")
            
                
    
    # Final synchronization and statistics
    dist.barrier()
    
    if rank == 0:
        total_time = sum(load_times)
        avg_load_time = total_time / len(load_times) if load_times else 0
        successful_loads = len(load_times)
        
        print(f"\n=== Load Test Results ===")
        print(f"Model size: {model_size}")
        print(f"Total iterations: {num_iterations}")
        print(f"Successful loads: {successful_loads}")
        print(f"Failed loads: {failed_loads}")
        print(f"Average load time: {avg_load_time:.3f}s")
        print(f"Total load time: {total_time:.3f}s")
        print(f"Effective load rate: {successful_loads / total_time:.3f} loads/second")
        
        if load_times:
            print(f"Min load time: {min(load_times):.3f}s")
            print(f"Max load time: {max(load_times):.3f}s")
        
        # Calculate approximate S3 request rate
        # This is a rough estimate - actual requests depend on checkpoint structure
        estimated_requests_per_load = world_size * 10  # Rough estimate
        total_s3_requests = successful_loads * estimated_requests_per_load
        s3_request_rate = total_s3_requests / total_time if total_time > 0 else 0
        
        print(f"Estimated S3 requests: {total_s3_requests}")
        print(f"Estimated S3 request rate: {s3_request_rate:.1f} req/sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"])
    parser.add_argument("--thread_count", type=int, default=1)
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--uri", type=str, required=True)
    
    # New arguments for repeated loading
    parser.add_argument("--iterations", type=int, default=5, 
                       help="Number of times to repeat the load operation")
    parser.add_argument("--delay", type=float, default=0.0,
                       help="Seconds to wait between load operations (0 for max throughput)")
    parser.add_argument("--concurrent", action="store_true",
                       help="Don't synchronize between ranks for maximum concurrency")
    parser.add_argument("--model", type=str, default="L7b",
                       help="Model name to use for benchmarking")
    
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
    suffix = "2025-06-23-12-32-SKnN"
    checkpoint_sharding_strategy = "hybrid"
    
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
        delay_between_loads=args.delay,
        concurrent_loads=args.concurrent
    )
    
    dist.barrier()
    dist.destroy_process_group()


