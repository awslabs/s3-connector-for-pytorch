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

import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from .constants import Timestamps
from .models import BenchmarkModel

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
    model_size_mib: float  # "_mib" == MiB


class Data(TypedDict):
    throughput_mibs: Any
    corrected_save_durations_s: Any
    processing_durations_s: Any


class Results(TypedDict):
    metadata: Metadata
    config: Any
    results: Data


# DataFrame column names
BEGIN_SAVE = "begin_save"
END_SAVE = "end_save"
BEGIN_PROCESS = "begin_process"
END_PROCESS = "end_process"


def save_results(
    cfg: DictConfig,
    model: BenchmarkModel,
    corrected_save_timestamps: List[Timestamps],
    processing_timestamps: List[Timestamps],
):
    """Save a Hydra job's results to a local JSON file."""

    cst = pd.DataFrame(corrected_save_timestamps, columns=[BEGIN_SAVE, END_SAVE])
    pt = pd.DataFrame(processing_timestamps, columns=[BEGIN_PROCESS, END_PROCESS])
    corrected_save_durations_s = cst[END_SAVE] - cst[BEGIN_SAVE]
    processing_durations_s = pt[END_PROCESS] - pt[BEGIN_PROCESS]
    throughput_mibs = model.size / corrected_save_durations_s

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
        "results": {
            "throughput_mibs": throughput_mibs.describe().to_dict(),
            "corrected_save_durations_s": corrected_save_durations_s.describe().to_dict(),
            "processing_durations_s": processing_durations_s.describe().to_dict(),
        },
    }

    # ["foo=4", "bar=small", "baz=1"] -> "4_small_1"
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
