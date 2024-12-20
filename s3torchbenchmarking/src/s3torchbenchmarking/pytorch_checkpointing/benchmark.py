# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: BSD

import logging
from pathlib import Path
from time import perf_counter
from typing import Dict, Any

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig

from s3torchbenchmarking.benchmark_utils import (
    ResourceMonitor,
    build_checkpoint_uri,
    build_random_suffix,
)
from s3torchbenchmarking.models import get_benchmark_model
from s3torchconnector import S3Checkpoint

logger = logging.getLogger(__name__)


def create_checkpoint_dir(path: str, suffix: str) -> Path:
    """Create and return the checkpoint directory."""
    parent_folder = Path(path) / suffix
    parent_folder.mkdir(parents=True, exist_ok=True)


def save_checkpoint(
    model: torch.nn.Module, path: str, checkpoint: S3Checkpoint = None
) -> float:
    """Save checkpoint and return the time taken."""
    start_time = perf_counter()
    if checkpoint:
        with checkpoint.writer(path) as writer:
            torch.save(model.state_dict(), writer)
    else:
        torch.save(model.state_dict(), path)
    end_time = perf_counter()
    return end_time - start_time


def calculate_metrics(
    save_times: list, model_size: float, monitor: ResourceMonitor
) -> Dict[str, Any]:
    """Calculate and return benchmark metrics."""
    save_times_s = pd.Series(save_times)
    throughput_mibs = model_size / save_times_s
    return {
        "throughput_mibs": throughput_mibs.dropna().tolist(),
        "save_times_s": save_times_s.dropna().tolist(),
        "utilization": {k: v.summarize() for k, v in monitor.resource_data.items()},
    }


@hydra.main(version_base=None)
def run_benchmark(config: DictConfig) -> Dict[str, Any]:
    """Checkpoint benchmarks entry point."""
    logger.info("Starting Checkpoint benchmark run")

    try:
        benchmark_model = get_benchmark_model(config.model)
        checkpoint = (
            None
            if config.checkpoint.storage == "disk"
            else S3Checkpoint(region=config.s3.region)
        )

        suffix = build_random_suffix()
        if config.checkpoint.storage == "disk":
            create_checkpoint_dir(config.path, suffix)

        save_times = []
        with ResourceMonitor() as monitor:
            for i in range(config.epochs):
                filepath = f"{suffix}/{config.model}-{i}.ckpt"
                if config.checkpoint.storage == "disk":
                    checkpoint_path = Path(config.path) / filepath
                else:
                    checkpoint_path = build_checkpoint_uri(config.s3.uri, filepath)

                logger.info(f"Saving checkpoint to {checkpoint_path}")
                save_time = save_checkpoint(
                    benchmark_model.model, str(checkpoint_path), checkpoint
                )
                save_times.append(save_time)

        metrics = calculate_metrics(save_times, benchmark_model.size, monitor)
        logger.info("Benchmark run completed successfully")
        return {"metrics": metrics}

    except Exception as e:
        logger.exception(f"An error occurred during the benchmark: {str(e)}")
        return {"error": str(e)}


if __name__ == "__main__":
    run_benchmark()
