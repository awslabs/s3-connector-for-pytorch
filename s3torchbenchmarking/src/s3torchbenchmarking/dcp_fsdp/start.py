#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import logging
import functools
from time import perf_counter
from typing import Tuple
import os
import argparse

import torch.distributed.checkpoint as dcp
import torch
import torch.distributed as dist
import torch.utils.data

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from torch.distributed.checkpoint import FileSystemWriter, FileSystemReader

from s3torchconnector.dcp import S3StorageWriter, S3StorageReader


from s3torchbenchmarking.models import get_benchmark_model


from s3torchbenchmarking.benchmark_utils import (
    build_random_suffix,
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

def setup(backend: str, world_size: int, rank: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    dist.init_process_group(backend, world_size=world_size, rank=rank)


def get_writer(region:str, uri: str, suffix: str, thread_count: int = 8) -> FileSystemWriter:
    uri = build_checkpoint_uri(uri, suffix)
    logger.info("Saving checkpoint to %s (S3)...", uri)
    return S3StorageWriter(region, uri, thread_count=thread_count)

def get_reader(region:str, uri: str, suffix: str) -> FileSystemReader:
    uri = build_checkpoint_uri(uri, suffix)
    logger.info("Loading checkpoint from %s (S3)...", uri)
    return S3StorageReader(region, uri)


def run_fsdp(
    rank: int,  # needs to be passed first (provided by `multiprocessing.spawn` automatically)
    world_size: int,
    thread_count: int,
    backend: str,
    region: str,
    uri: str,
    suffix: str,
    model_name: str = "L7b",
    checkpoint_sharding_strategy: str = "full"
) -> None:
    """Execute the actual code for checkpoint saving.

    This function is meant to be executed in subprocesses."""
    # setup(backend=backend, world_size=world_size, rank=rank)

    if rank == 0:
        logger.info("Creating Model")
    # Instantiate model on CPU on rank=0 only to prevent CPU OOM
    # (e.g. 70B * 4 bytes * 8 processes > 2T RAM available on P5)
    if rank == 0:
        model_proxy = get_benchmark_model(model_name)
        model = model_proxy.model
    else:
        with torch.device("meta"):
            # Instantiating model on `meta` device doesn't consume CPU memory,
            # but requires specifing `param_init_fn=...`
            # and `sync_module_states=True` in FSDP c-tor.
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

    # torch.cuda.empty_cache()
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        state_dict = {
            "model": model.state_dict(),
        }

    storage_writer = get_writer(region, uri, suffix, thread_count)
    # align all workers to start checkpointing at the same time
    dist.barrier()
    begin_save = perf_counter()
    dcp.save(state_dict, storage_writer=storage_writer)

    dist.barrier()
    end_save = perf_counter()

    if rank == 0:
        print(f"The total size of model is {model_size}")
        print(f"Time taken to save: {end_save - begin_save} seconds")
    # Record the save times excluding the influence of the process setup and model loading to device.

    storage_reader = get_reader(region, uri, suffix)
    # empty_stat_dict = {"model": None}
    start_load = perf_counter()
    dcp.load(state_dict, storage_reader=storage_reader)
    end_load = perf_counter()

    if rank == 0:
        print(f"Time taken to load: {end_load - start_load} seconds")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"])
    parser.add_argument("--thread_count", type=int, default=1)
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--uri", type=str)
    args = parser.parse_args()

    backend = args.backend
    dist.init_process_group(backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Starting for rank {rank}, world_size is {world_size}")
    thread_count = args.thread_count

    region = args.region
    uri = args.uri
    suffix = "experiment"
    checkpoint_sharding_strategy = "hybrid"
    run_fsdp(rank, world_size, thread_count, backend, region, uri, suffix, checkpoint_sharding_strategy=checkpoint_sharding_strategy)