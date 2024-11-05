#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import logging
import os
import random
import string
from os import PathLike
from pathlib import Path
from time import perf_counter
from typing import Union

import hydra
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch import multiprocessing
from torch.distributed.checkpoint import FileSystemWriter
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel

from s3torchbenchmarking.benchmark_utils import ResourceMonitor
from s3torchbenchmarking.dcp.models import get_benchmark_model_from_size
from s3torchbenchmarking.dcp.results import save_results
from s3torchbenchmarking.dcp.distribution import Distribution
from s3torchconnector.dcp import S3StorageWriter

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=".", config_name="config")
def run_benchmark(cfg: DictConfig):
    """DCP benchmark entry point."""
    logger.info("Saving results to: %s", HydraConfig.get().runtime.output_dir)

    benchmark_model = get_benchmark_model_from_size(cfg.model)
    world_size = validate_world_size(cfg.world_size)

    # For every run, use a randomized suffix (for either S3 or local storage).
    suffix = "".join(random.choices(string.ascii_letters, k=7))

    storage_writer = get_writer(
        region=cfg.s3.region,
        path=Path(cfg.path) / suffix,
        s3_uri=build_checkpoint_uri(cfg.s3.uri, suffix),
        thread_count=cfg.thread_count,
        storage=cfg.checkpoint.storage,
    )

    save_times = Distribution()
    with ResourceMonitor() as monitor:
        for epoch in range(cfg.epochs):
            logger.info("Running epoch #%i / %i...", epoch + 1, cfg.epochs)
            start_time = perf_counter()
            multiprocessing.spawn(
                run,
                args=(
                    cfg.backend,
                    benchmark_model.pre_trained_model,
                    world_size,
                    storage_writer,
                ),
                nprocs=world_size,
                join=True,
            )
            elapsed_time = perf_counter() - start_time
            save_times.append(elapsed_time)

    save_results(cfg, benchmark_model, save_times, monitor)


def validate_world_size(world_size: int) -> int:
    """Enforce `world_size` to be within the current node's capacity."""

    # FIXME: only works when the backend is "nccl"; what about "gloo" and CPUs?
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


def get_writer(
    region: str,
    path: Union[str, PathLike],
    s3_uri: str,
    thread_count: int,
    storage: str,
) -> FileSystemWriter:
    if storage == "disk":
        return dcp.FileSystemWriter(path, thread_count=thread_count)
    elif storage == "s3":
        return S3StorageWriter(
            region,
            s3_uri,
            thread_count=thread_count,
        )
    raise ValueError(f"Storage writer {storage} not supported")


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
    model: Module,
    world_size: int,
    storage_writer: FileSystemWriter,
) -> None:
    logger.info("Running on rank %i...", rank)

    torch.cuda.set_device(rank)
    setup(backend=backend, world_size=world_size, rank=rank)
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()

    logger.debug("Device ID: %i", device_id)

    model.to(device_id)
    wrapped_model = DistributedDataParallel(model, device_ids=[device_id])

    logger.info("Saving checkpoint on rank %i...", rank)
    start_time = perf_counter()
    dcp.save(
        wrapped_model.state_dict(),
        storage_writer=storage_writer,
    )
    elapsed_time = perf_counter() - start_time
    logger.debug("DCP save time: %f s", elapsed_time)
    logger.info("Checkpoint successfully saved on rank %i", rank)

    dist.destroy_process_group()
