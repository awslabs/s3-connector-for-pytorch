#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import logging
import functools
from multiprocessing.queues import Queue
from pathlib import Path
from time import perf_counter
from typing import List, Tuple

import hydra
import pandas as pd
import torch.distributed.checkpoint as dcp
from omegaconf import DictConfig
from torch import multiprocessing as mp
from torch.distributed.checkpoint import FileSystemWriter

import os
import torch
import torch.distributed as dist
import torch.utils.data

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from s3torchbenchmarking.benchmark_utils import (
    build_random_suffix,
    build_checkpoint_uri,
)
from s3torchbenchmarking.models import get_benchmark_model
from s3torchconnector.dcp import S3StorageWriter

Timestamps = Tuple[float, float]
logger = logging.getLogger(__name__)


# TODO: add Structured Config (https://hydra.cc/docs/tutorials/structured_config/intro/)
@hydra.main(version_base=None)
def run_benchmark(cfg: DictConfig) -> dict:
    """DCP benchmarks entry point."""
    manager = mp.Manager()
    corrected_save_timestamps: Queue[Timestamps] = manager.Queue()
    processing_timestamps: List[Timestamps] = []

    suffix = build_random_suffix()

    for epoch in range(cfg.epochs):
        logger.info("Executing epoch #%i / %i...", epoch + 1, cfg.epochs)
        begin_mp = perf_counter()
        mp.spawn(
            _run,
            (cfg, suffix, corrected_save_timestamps),
            nprocs=cfg.world_size,
            join=True,
        )
        end_mp = perf_counter()
        processing_timestamps.append((begin_mp, end_mp))

    # Dump the multiprocessing Queue's content into a list.
    collector: List[Timestamps] = []
    while not corrected_save_timestamps.empty():
        collector.append(corrected_save_timestamps.get())

    # Collect all data in Pandas DataFrame, and dump them (through `describe()`) in a Python dict.
    cst = pd.DataFrame(collector, columns=["begin", "end", "size"])
    pt = pd.DataFrame(processing_timestamps, columns=["begin", "end"])

    corrected_save_durations_s = cst["end"] - cst["begin"]
    processing_durations_s = pt["end"] - pt["begin"]
    # Expectation that all workers started at the same time. So we are taking the longest time
    # to get average throughput for the entire run.
    throughput_mibs = cst["size"] / corrected_save_durations_s.max()

    metrics = {
        "throughput_mibs": throughput_mibs.dropna().to_list(),
        "corrected_save_durations_s": corrected_save_durations_s.dropna().to_list(),
        "processing_durations_s": processing_durations_s.dropna().to_list(),
    }
    return {"metrics": metrics}


def get_writer(cfg: DictConfig, suffix) -> FileSystemWriter:
    """Instantiate a checkpoint writer based on the input config."""

    if cfg.checkpoint.storage == "disk":
        local_path = Path(cfg.path) / suffix
        logger.info("Saving checkpoint to %s (disk)...", local_path)
        sync_files = (
            True
            if cfg.checkpoint.sync_files is not None and cfg.checkpoint.sync_files
            else False
        )
        return dcp.FileSystemWriter(
            local_path, sync_files=sync_files, thread_count=cfg.thread_count
        )
    elif cfg.checkpoint.storage == "s3":
        uri = build_checkpoint_uri(cfg.s3.uri, suffix)
        logger.info("Saving checkpoint to %s (S3)...", uri)
        return S3StorageWriter(cfg.s3.region, uri, thread_count=cfg.thread_count)
    else:
        raise ValueError(f"Storage writer {cfg.checkpoint.storage} not supported")


def setup(backend: str, world_size: int, rank: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    dist.init_process_group(backend, world_size=world_size, rank=rank)


def _run(
    rank: int,  # needs to be passed first (provided by `multiprocessing.spawn` automatically)
    cfg: DictConfig,
    suffix,
    save_timestamps: Queue,
) -> None:
    """Execute the actual code for checkpoint saving.

    This function is meant to be executed in subprocesses."""
    setup(cfg.backend, world_size=cfg.world_size, rank=rank)

    if rank == 0:
        logger.info("Creating Model")
    # Instantiate model on CPU on rank=0 only to prevent CPU OOM
    # (e.g. 70B * 4 bytes * 8 processes > 2T RAM available on P5)
    if rank == 0:
        model_proxy = get_benchmark_model(cfg.model)
        model = model_proxy.model
    else:
        with torch.device("meta"):
            # Instantiating model on `meta` device doesn't consume CPU memory,
            # but requires specifing `param_init_fn=...`
            # and `sync_module_states=True` in FSDP c-tor.
            model_proxy = get_benchmark_model(cfg.model)
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

    if cfg.backend == "nccl":
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

    if cfg.checkpoint.sharding_strategy == "full":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif cfg.checkpoint.sharding_strategy == "hybrid":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError("Available sharding strategies are full and hybrid")

    model = FSDP(
        model,
        auto_wrap_policy=gpt_auto_wrap_policy,
        device_id=(
            torch.cuda.current_device()
            if cfg.backend == "nccl"
            else torch.cpu.current_device()
        ),
        use_orig_params=False,
        sharding_strategy=sharding_strategy,
        sync_module_states=True if cfg.backend == "nccl" else False,
        param_init_fn=param_init_fn if rank != 0 else None,
    )

    if rank == 0:
        logger.info("Wrapped model with FSDP")

    # torch.cuda.empty_cache()
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        state_dict = {
            "model": model.state_dict(),
        }

    storage_writer = get_writer(cfg, suffix)
    # align all workers to start checkpointing at the same time
    dist.barrier()
    begin_save = perf_counter()
    dcp.save(state_dict, storage_writer=storage_writer)
    end_save = perf_counter()

    if rank == 0:
        logger.info(f"The total size of model is {model_size}")
    # Record the save times excluding the influence of the process setup and model loading to device.
    save_timestamps.put((begin_save, end_save, model_size))
    dist.destroy_process_group()


if __name__ == "__main__":
    run_benchmark()
