#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import logging
import os
import random
import string
from multiprocessing.queues import Queue
from pathlib import Path
from time import perf_counter
from typing import List

import hydra
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from omegaconf import DictConfig
from torch import multiprocessing as mp
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.fsdp import FullyShardedDataParallel, StateDictType
from torch.nn import Module

from s3torchconnector.dcp import S3StorageWriter
from .constants import Timestamps
from .models import get_benchmark_model
from .results import save_results
from ..benchmark_utils import ResourceMonitor

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=".", config_name="config")
def run_benchmark(cfg: DictConfig):
    """DCP benchmark entry point."""
    benchmark_model = get_benchmark_model(cfg.model)

    # For every run, use a randomized suffix (for either local disk or S3).
    suffix = "".join(random.choices(string.ascii_letters, k=7))
    storage_writer = get_writer(cfg, suffix)

    manager = mp.Manager()
    corrected_save_timestamps: Queue[Timestamps] = manager.Queue()
    processing_timestamps: List[Timestamps] = []

    with ResourceMonitor() as monitor:
        for epoch in range(cfg.epochs):
            begin_mp = perf_counter()
            mp.spawn(
                run,
                (cfg, benchmark_model.model, storage_writer, corrected_save_timestamps),
                nprocs=cfg.world_size,
                join=True,
            )
            end_mp = perf_counter()
            processing_timestamps.append((begin_mp, end_mp))

    # Dump the multiprocessing Queue's content into a list.
    collector: List[Timestamps] = []
    while not corrected_save_timestamps.empty():
        collector.append(corrected_save_timestamps.get())

    save_results(
        cfg,
        benchmark_model,
        corrected_save_timestamps=collector,
        processing_timestamps=processing_timestamps,
    )


def get_writer(cfg: DictConfig, suffix: str) -> FileSystemWriter:
    """Instantiate a checkpoint writer based on the input config."""
    if cfg.checkpoint.storage == "disk":
        local_path = Path(cfg.path) / suffix
        logger.info("Saving checkpoint to %s (local disk)...", local_path)
        return dcp.FileSystemWriter(local_path, thread_count=cfg.thread_count)
    elif cfg.checkpoint.storage == "s3":
        uri = build_checkpoint_uri(cfg.s3.uri, suffix)
        logger.info("Saving checkpoint to %s (S3)...", uri)
        return S3StorageWriter(cfg.s3.region, uri, thread_count=cfg.thread_count)
    raise ValueError(f"Storage writer {cfg.checkpoint.storage} not supported")


def build_checkpoint_uri(s3_uri: str, suffix: str) -> str:
    suffix = suffix.lstrip("/")
    return s3_uri + suffix if s3_uri.endswith("/") else s3_uri + "/" + suffix


def setup(backend: str, world_size: int, rank: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend, world_size=world_size, rank=rank)


# FIXME: configure logging in subprocess accordingly
def run(
    rank: int,  # needs to be passed first (provided by `multiprocessing.spawn` automatically)
    cfg: DictConfig,
    model: Module,
    storage_writer: FileSystemWriter,
    save_timestamps: Queue,
) -> None:
    """Execute the actual code for checkpoint saving.

    This function is meant to be executed in subprocesses."""
    begin_process = perf_counter()

    setup(cfg.backend, world_size=cfg.world_size, rank=rank)
    if cfg.backend == "nccl":
        device_id = rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
    else:
        device_id = rank % torch.cpu.device_count()
        torch.cpu.set_device(device_id)

    FullyShardedDataParallel.set_state_dict_type(
        model, StateDictType.SHARDED_STATE_DICT
    )
    fsdp_model = FullyShardedDataParallel(model, device_id=torch.cuda.current_device())

    begin_save = perf_counter()
    dcp.save(fsdp_model.state_dict(), storage_writer=storage_writer)
    end_save = perf_counter()

    # Record the save times excluding the influence of the process setup and model loading to device.
    save_timestamps.put((begin_process, end_save - (begin_save - begin_process)))

    dist.destroy_process_group()
