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
from torch import multiprocessing
from torch.distributed.checkpoint import FileSystemWriter
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel

from s3torchbenchmarking.benchmark_utils import ResourceMonitor
from s3torchbenchmarking.dcp.distribution import Distribution
from s3torchbenchmarking.dcp.models import get_benchmark_model
from s3torchbenchmarking.dcp.results import save_results
from s3torchconnector.dcp import S3StorageWriter

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=".", config_name="config")
def run_benchmark(cfg: DictConfig):
    """DCP benchmark entry point."""
    benchmark_model = get_benchmark_model(cfg.model)

    # For every run, use a randomized suffix (for either local disk or S3).
    suffix = "".join(random.choices(string.ascii_letters, k=7))
    storage_writer = get_writer(cfg, suffix)

    manager = multiprocessing.Manager()
    save_durations: Queue[float] = manager.Queue()

    with ResourceMonitor() as monitor:
        start_time = perf_counter()
        multiprocessing.spawn(
            run,
            (cfg, benchmark_model.model, storage_writer, save_durations),
            nprocs=cfg.world_size,
            join=True,
        )
        end_time = perf_counter()

    processing_durations = Distribution([end_time - start_time])

    # Dump the multiprocessing Queue's content into a list.
    collector: List[float] = []
    while not save_durations.empty():
        collector.append(save_durations.get())
    save_durations_distribution = Distribution(collector)

    save_results(
        cfg,
        benchmark_model,
        save_durations=save_durations_distribution,
        processing_durations=processing_durations,
        monitor=monitor,
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
    save_durations: Queue,
) -> None:
    """Execute the actual code for checkpoint saving.

    This function is meant to be executed in subprocesses."""

    setup(cfg.backend, world_size=cfg.world_size, rank=rank)
    rank = dist.get_rank()
    if cfg.backend == "nccl":
        device_id = rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
    else:
        device_id = rank % torch.cpu.device_count()
        torch.cpu.set_device(device_id)

    model.to(device_id)
    ddp_model = DistributedDataParallel(model, device_ids=[device_id])

    start_time = perf_counter()
    for i in range(cfg.epochs):
        dcp.save(ddp_model.state_dict(), storage_writer=storage_writer)
    end_time = perf_counter()

    save_durations.put(end_time - start_time)

    dist.destroy_process_group()
