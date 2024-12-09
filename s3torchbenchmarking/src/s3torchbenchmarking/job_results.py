#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import json
import logging
from pathlib import Path
from typing import Any

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from s3torchbenchmarking.constants import JOB_RESULTS_FILENAME
from s3torchbenchmarking.models import BenchmarkModel

logger = logging.getLogger(__name__)


def save_job_results(cfg: DictConfig, model: BenchmarkModel, metrics: Any) -> None:
    """Save a single Hydra job results to a JSON file."""
    results = {
        "model": {
            "name": model.name,
            "size_mib": model.size,
        },
        "config": OmegaConf.to_container(cfg),
        "metrics": metrics,
    }

    results_dir = HydraConfig.get().runtime.output_dir
    results_path = Path(results_dir, JOB_RESULTS_FILENAME)

    logger.info("Saving job results to: %s", results_path)
    with open(results_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Job results saved successfully")
