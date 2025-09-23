#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import logging
from multiprocessing.queues import Queue
from time import perf_counter
from typing import Tuple

import hydra
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel

from s3torchbenchmarking.dcp_common import setup, get_reader, benchmark_common_runner
from s3torchbenchmarking.models import get_benchmark_model, BenchmarkModel

Timestamps = Tuple[float, float]
logger = logging.getLogger(__name__)


# TODO: add Structured Config (https://hydra.cc/docs/tutorials/structured_config/intro/)
@hydra.main(version_base=None)
def run_benchmark(cfg: DictConfig) -> dict:
    """DCP benchmarks entry point."""
    benchmark_model = get_benchmark_model(cfg.model)

    return benchmark_common_runner(cfg, run_ddp_load, (cfg, benchmark_model))


def run_ddp_load(
    rank: int,  # needs to be passed first (provided by `multiprocessing.spawn` automatically)
    cfg: DictConfig,
    proxy_model: BenchmarkModel,
    suffix: str,
    load_timestamps: Queue,
) -> None:
    """Execute the actual code for checkpoint loading.

    This function is meant to be executed in subprocesses."""
    begin_process = perf_counter()
    # Override random suffix with suffix from config
    storage_reader = get_reader(cfg)
    model_size = proxy_model.size
    model = proxy_model.model

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

    begin_load = perf_counter()  # also "end_process"
    dcp.load(state_dict, storage_reader=storage_reader)
    end_load = perf_counter()

    # Record the load times excluding the influence of the process setup and model loading to device.
    load_timestamps.put(
        (begin_process, end_load - (begin_load - begin_process), model_size)
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    run_benchmark()
