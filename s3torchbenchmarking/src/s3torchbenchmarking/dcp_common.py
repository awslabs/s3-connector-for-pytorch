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
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from omegaconf import DictConfig
from torch import multiprocessing as mp
from torch.distributed.checkpoint import FileSystemWriter, FileSystemReader

from s3torchbenchmarking.benchmark_utils import (
    build_random_suffix,
    build_checkpoint_uri,
)
from s3torchconnector.dcp import S3StorageWriter, S3StorageReader

Timestamps = Tuple[float, float]
logger = logging.getLogger(__name__)


def setup(backend: str, world_size: int, rank: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    dist.init_process_group(backend, world_size=world_size, rank=rank)


def get_writer(cfg: DictConfig, suffix: str) -> FileSystemWriter:
    """Instantiate a checkpoint writer based on the input config."""
    if cfg.checkpoint.storage == "disk":
        local_path = Path(cfg.path) / suffix
        logger.info("Saving checkpoint to %s (disk)...", local_path)
        sync_files = getattr(cfg.checkpoint, "sync_files", False)
        return dcp.FileSystemWriter(
            local_path, sync_files=sync_files, thread_count=cfg.thread_count
        )
    elif cfg.checkpoint.storage == "s3":
        uri = build_checkpoint_uri(cfg.s3.uri, suffix)
        logger.info("Saving checkpoint to %s (S3)...", uri)
        return S3StorageWriter(cfg.s3.region, uri, thread_count=cfg.thread_count, num_copies =cfg.num_of_copies)
    raise ValueError(f"Storage writer {cfg.checkpoint.storage} not supported")


def get_reader(cfg: DictConfig) -> FileSystemReader:
    """Instantiate a checkpoint reader based on the input config."""
    suffix = cfg.checkpoint.suffix
    if cfg.checkpoint.storage == "disk":
        local_path = Path(cfg.path) / suffix
        logger.info("Loading checkpoint from %s (disk)...", local_path)
        return dcp.FileSystemReader(local_path)
    elif cfg.checkpoint.storage == "s3":
        uri = build_checkpoint_uri(cfg.s3.uri, suffix)
        logger.info("Loading checkpoint from %s (S3)...", uri)
        return S3StorageReader(cfg.s3.region, uri)
    raise ValueError(f"Storage reader {cfg.checkpoint.storage} not supported")


def benchmark_common_runner(
    cfg: DictConfig,
    run_fn,
    run_args: tuple,
) -> dict:
    manager = mp.Manager()
    corrected_save_timestamps: Queue[Timestamps] = manager.Queue()
    processing_timestamps: List[Timestamps] = []

    for epoch in range(cfg.epochs):
        suffix = build_random_suffix()
        logger.info("Executing epoch #%i / %i...", epoch + 1, cfg.epochs)
        begin_mp = perf_counter()
        mp.spawn(
            run_fn,
            run_args
            + (
                suffix,
                corrected_save_timestamps,
            ),
            nprocs=cfg.world_size,
            join=True,
        )
        end_mp = perf_counter()
        processing_timestamps.append((begin_mp, end_mp))

    return process_timestamps(corrected_save_timestamps, processing_timestamps)


def process_timestamps(
    corrected_save_timestamps: Queue,
    processing_timestamps: List[Timestamps],
) -> dict:
    """Process and return metrics from timestamps."""
    collector: List[Timestamps] = []
    while not corrected_save_timestamps.empty():
        collector.append(corrected_save_timestamps.get())

    cst = pd.DataFrame(collector, columns=["begin", "end", "size"])
    pt = pd.DataFrame(processing_timestamps, columns=["begin", "end"])

    corrected_save_durations_s = cst["end"] - cst["begin"]
    processing_durations_s = pt["end"] - pt["begin"]
    throughput_mibs = cst["size"] / corrected_save_durations_s

    return {
        "metrics": {
            "throughput_mibs": throughput_mibs.dropna().to_list(),
            "corrected_save_durations_s": corrected_save_durations_s.dropna().to_list(),
            "processing_durations_s": processing_durations_s.dropna().to_list(),
        }
    }
