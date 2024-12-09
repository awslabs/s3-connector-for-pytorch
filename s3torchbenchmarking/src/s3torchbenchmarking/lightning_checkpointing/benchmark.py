#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import logging
from pathlib import Path

import hydra
import pandas as pd
from lightning import Trainer
from lightning.pytorch import callbacks
from lightning.pytorch.strategies import SingleDeviceStrategy
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper  # type: ignore

from s3torchbenchmarking.benchmark_utils import (
    ResourceMonitor,
    build_checkpoint_uri,
    build_random_suffix,
)
from s3torchbenchmarking.job_results import save_job_results
from s3torchbenchmarking.lightning_checkpointing.checkpoint_profiler import (
    CheckpointProfiler,
)
from s3torchbenchmarking.models import get_benchmark_model, LightningAdapter
from s3torchconnector.lightning import S3LightningCheckpoint

logger = logging.getLogger(__name__)


@hydra.main(version_base=None)
def run_benchmark(config: DictConfig):
    """Lightning benchmarks entry point."""
    benchmark_model = get_benchmark_model(config.model)

    strategy = SingleDeviceStrategy()

    if config.checkpoint.storage == "disk":
        checkpoint_callback = callbacks.ModelCheckpoint(dirpath=config.path)
        checkpoint_io = strategy.checkpoint_io
    else:
        checkpoint_callback = callbacks.ModelCheckpoint(dirpath=config.s3.uri)
        checkpoint_io = S3LightningCheckpoint(config.s3.region)

    profiling_checkpointer = CheckpointProfiler(checkpoint_io)
    trainer = Trainer(
        logger=False, plugins=[profiling_checkpointer], callbacks=[checkpoint_callback]
    )
    dataloader = DataLoader(IterableWrapper([]), num_workers=8)
    trainer.fit(
        LightningAdapter.DelegateModule(benchmark_model.model),
        train_dataloaders=dataloader,
    )

    suffix = build_random_suffix()
    with ResourceMonitor() as monitor:
        for i in range(config.epochs):
            filepath = f"{suffix}/{config.model}-{i}.ckpt"
            if config.checkpoint.storage == "disk":
                checkpoint_path = Path(config.path) / filepath
            else:
                checkpoint_path = build_checkpoint_uri(config.s3.uri, filepath)
            trainer.save_checkpoint(checkpoint_path)

    save_times_s = pd.Series(profiling_checkpointer.save_times)
    throughput_mibs = benchmark_model.size / save_times_s

    metrics = {
        "throughput_mibs": throughput_mibs.describe().to_dict(),
        "save_times_s": save_times_s.describe().to_dict(),
        "utilization": {k: v.summarize() for k, v in monitor.resource_data.items()},
    }

    save_job_results(config, benchmark_model, metrics)


if __name__ == "__main__":
    run_benchmark()
