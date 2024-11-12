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
import numpy as np
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
    world_size = validate_world_size(cfg.world_size)

    # For every run, use a randomized suffix (for either local disk or S3).
    suffix = "".join(random.choices(string.ascii_letters, k=7))
    storage_writer = get_writer(cfg, suffix)

    manager = multiprocessing.Manager()
    save_timestamps: Queue[float] = manager.Queue()

    processing_times = Distribution()
    ptp_save_times = Distribution()  # "ptp" == "peak to peak", or "diff(max, min)"

    with ResourceMonitor() as monitor:
        start_time = perf_counter()
        multiprocessing.spawn(
            run,
            (
                cfg.backend,
                cfg.epochs,
                benchmark_model.model,
                world_size,
                storage_writer,
                save_timestamps,
            ),
            nprocs=world_size,
            join=True,
        )
        end_time = perf_counter()
        processing_times.append(end_time - start_time)

        collector: List[float] = []
        while not save_timestamps.empty():
            collector.append(save_timestamps.get())
        ptp_save_times.append(np.ptp(collector))

    save_results(
        cfg,
        benchmark_model,
        ptp_save_times=ptp_save_times,
        processing_times=processing_times,
        monitor=monitor,
    )


def validate_world_size(world_size: int) -> int:
    """Enforce `world_size` to be within the current node's capacity.

    FIXME: only works when the backend is "nccl"; what about "gloo" and CPUs?
    """
    device_count = torch.cuda.device_count()
    if world_size > device_count:
        logger.warning(
            "Received a `world_size` of %i, while only %i are available: decreasing to %i",
            world_size,
            device_count,
            device_count,
        )
        return device_count
    return world_size


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
    dist.init_process_group(backend=backend, world_size=world_size, rank=rank)


def run(
    rank: int,  # needs to be passed first (provided by `multiprocessing.spawn` automatically)
    backend: str,
    epochs: int,
    model: Module,
    world_size: int,
    storage_writer: FileSystemWriter,
    save_timestamps: Queue,
) -> None:
    """Execute the actual code for checkpoint saving.

    This function is meant to be executed in subprocesses."""

    # FIXME: configure logging in subprocess accordingly

    torch.cuda.set_device(rank)
    setup(backend=backend, world_size=world_size, rank=rank)
    rank = dist.get_rank()

    device_id = rank % torch.cuda.device_count()
    model.to(device_id)

    ddp_model = DistributedDataParallel(model, device_ids=[device_id])

    for i in range(epochs):
        start_time = perf_counter()
        dcp.save(ddp_model.state_dict(), storage_writer=storage_writer)
        end_time = perf_counter()

        save_timestamps.put(start_time)
        save_timestamps.put(end_time)

    dist.destroy_process_group()
