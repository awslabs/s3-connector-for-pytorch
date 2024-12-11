#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import logging
import os
from multiprocessing.queues import Queue
from pathlib import Path
from time import perf_counter
from typing import List, Tuple

import hydra
import pandas as pd
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from omegaconf import DictConfig
from torch import multiprocessing as mp
from torch.distributed.checkpoint import FileSystemWriter
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel

from s3torchbenchmarking.benchmark_utils import (
    build_random_suffix,
    build_checkpoint_uri,
)
from s3torchbenchmarking.job_results import save_job_results
from s3torchbenchmarking.models import get_benchmark_model
from s3torchconnector.dcp import S3StorageWriter

Timestamps = Tuple[float, float]
logger = logging.getLogger(__name__)


@hydra.main(version_base=None)
def run_benchmark(cfg: DictConfig) -> None:
    """DCP benchmark entry point."""
    benchmark_model = get_benchmark_model(cfg.model)

    storage_writer = get_writer(cfg)

    manager = mp.Manager()
    corrected_save_timestamps: Queue[Timestamps] = manager.Queue()
    processing_timestamps: List[Timestamps] = []

    for epoch in range(cfg.epochs):
        logger.info("Executing epoch #%i / %i...", epoch + 1, cfg.epochs)
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

    # Collect all data in Pandas DataFrame, and dump them (through `describe()`) in a Python dict.
    cst = pd.DataFrame(collector, columns=["begin", "end"])
    pt = pd.DataFrame(processing_timestamps, columns=["begin", "end"])

    corrected_save_durations_s = cst["end"] - cst["begin"]
    processing_durations_s = pt["end"] - pt["begin"]
    throughput_mibs = benchmark_model.size / corrected_save_durations_s

    metrics = {
        "throughput_mibs": throughput_mibs.describe().to_dict(),
        "corrected_save_durations_s": corrected_save_durations_s.describe().to_dict(),
        "processing_durations_s": processing_durations_s.describe().to_dict(),
    }

    save_job_results(cfg, benchmark_model, metrics)


def get_writer(cfg: DictConfig) -> FileSystemWriter:
    """Instantiate a checkpoint writer based on the input config."""
    suffix = build_random_suffix()

    if cfg.checkpoint.storage == "disk":
        local_path = Path(cfg.path) / suffix
        logger.info("Saving checkpoint to %s (disk)...", local_path)
        return dcp.FileSystemWriter(local_path, thread_count=cfg.thread_count)
    elif cfg.checkpoint.storage == "s3":
        uri = build_checkpoint_uri(cfg.s3.uri, suffix)
        logger.info("Saving checkpoint to %s (S3)...", uri)
        return S3StorageWriter(cfg.s3.region, uri, thread_count=cfg.thread_count)
    raise ValueError(f"Storage writer {cfg.checkpoint.storage} not supported")


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
        model.to(device_id)
        model = DistributedDataParallel(model, device_ids=[device_id])
    else:
        device_id = rank % torch.cpu.device_count()
        torch.cpu.set_device(device_id)
        model.to(device=torch.device("cpu"))
        model = DistributedDataParallel(model)

    state_dict = model.state_dict()

    begin_save = perf_counter()  # also "end_process"
    dcp.save(state_dict, storage_writer=storage_writer)
    end_save = perf_counter()

    # Record the save times excluding the influence of the process setup and model loading to device.
    save_timestamps.put((begin_process, end_save - (begin_save - begin_process)))

    dist.destroy_process_group()


if __name__ == "__main__":
    run_benchmark()
