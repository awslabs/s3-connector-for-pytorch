#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import json
import logging
from pathlib import Path
from typing import Any

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from .models import BenchmarkModel

logger = logging.getLogger(__name__)


def save_job_results(
    cfg: DictConfig,
    model: BenchmarkModel,
    metrics: Any,
):
    """Save a Hydra job results to a local JSON file."""

    results = {
        "model": {
            "name": model.name,
            "size_mib": model.size,
        },
        "config": OmegaConf.to_container(cfg),
        "metrics": metrics,
    }

    # ["foo=4", "bar=small", "baz=1"] -> "4_small_1"
    tasks = HydraConfig.get().overrides.task
    # extract only sweeper values (i.e., ones starting with '+')
    tasks = [task for task in tasks if task.startswith("+")]
    suffix = "_".join([task.split("=")[-1] for task in tasks]) if tasks else ""

    # Save the results in the corresponding Hydra job directory (e.g., multirun/2024-11-08/15-47-08/0/<filename>.json).
    results_filename = f"results{'_' + suffix if suffix else ''}.json"
    results_dir = HydraConfig.get().runtime.output_dir
    results_path = Path(results_dir, results_filename)

    logger.info("Saving job results to: %s", results_path)
    with open(results_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    logger.info("Job results saved successfully")
