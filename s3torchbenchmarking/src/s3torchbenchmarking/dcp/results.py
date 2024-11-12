#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import json
import logging
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import TypedDict, Union, Any, List

import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from s3torchbenchmarking.benchmark_utils import ResourceMonitor
from s3torchbenchmarking.dcp.distribution import Distribution, Statistics
from s3torchbenchmarking.dcp.models import BenchmarkModel

logger = logging.getLogger(__name__)


class EC2Metadata(TypedDict):
    instance_type: str
    placement: str


class Metadata(TypedDict):
    python_version: str
    pytorch_version: str
    hydra_version: str
    ec2_metadata: Union[EC2Metadata, None]
    model_name: str
    model_size_mib: float  # "_mib" stands for MiB


class Data(TypedDict):
    throughput: Statistics
    save_durations: Statistics
    processing_durations: Statistics
    utilization: dict


class Results(TypedDict):
    metadata: Metadata
    config: Any
    data: Data


def save_results(
    cfg: DictConfig,
    model: BenchmarkModel,
    save_durations: Distribution,
    processing_durations: Distribution,
    monitor: ResourceMonitor,
):
    """Save a Hydra job's results to a local JSON file."""

    results: Results = {
        "metadata": {
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "hydra_version": HydraConfig.get().runtime.version,
            "ec2_metadata": get_ec2_metadata(),
            "model_name": model.name,
            "model_size_mib": model.size,
        },
        "config": OmegaConf.to_container(cfg),
        "data": {
            "throughput": to_throughput(save_durations, model.size).dump(unit="MiB/s"),
            "save_durations": save_durations.dump(unit="s"),
            "processing_durations": processing_durations.dump(unit="s"),
            "utilization": {k: v.summarize() for k, v in monitor.resource_data.items()},
        },
    }

    # `tasks` will contain the list of Hydra overrides (defined in the config.yaml file) in the form `"param=value"`;
    # this helps identify result files uniquely just by their filenames.
    # E.g.: `["foo=4", "bar=small", "baz=1"]` -> `suffix == "4_small_1"`.
    tasks = HydraConfig.get().overrides.task
    suffix = "_".join([task.split("=")[-1] for task in tasks])

    # Save the results in the corresponding Hydra job directory (e.g., multirun/2024-11-08/15-47-08/0/<filename>.json).
    results_filename = f"results{'_' + suffix if suffix else ''}.json"
    results_dir = HydraConfig.get().runtime.output_dir
    results_path = Path(results_dir, results_filename)

    logger.info("Saving results to: %s", results_path)
    with open(results_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    logger.info("Results saved successfully")


def to_throughput(save_times: List[float], model_size: float) -> Distribution:
    """Compute throughput from save times, in MiB/s."""
    return Distribution(map(lambda x: model_size / x, save_times))


@lru_cache
def get_ec2_metadata() -> Union[EC2Metadata, None]:
    """Get some EC2 metadata by running the `/opt/aws/bin/ec2-metadata` command.

    The command's output is a single string of text, in a JSON-like format (_but not quite JSON_): hence, its content
    is parsed using regex.

    The function's call is cached, so we don't execute the command multiple times per runs.
    """
    result = subprocess.run(
        "/opt/aws/bin/ec2-metadata", capture_output=True, text=True, timeout=5
    )
    if result.returncode == 0:
        metadata = result.stdout
        instance_type = re.search("instance-type: (.*)", metadata).group(1)
        placement = re.search("placement: (.*)", metadata).group(1)
        if instance_type and placement:
            return {"instance_type": instance_type, "placement": placement}
    return None
