import json
import logging
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any, List, TypedDict, Union, Optional

import torch
from hydra.experimental.callback import Callback
from omegaconf import DictConfig

_COLLATED_RESULTS_FILENAME = "collated_results.json"

logger = logging.getLogger(__name__)


class EC2Metadata(TypedDict):
    instance_type: str
    placement: str


class Metadata(TypedDict):
    python_version: str
    pytorch_version: str
    hydra_version: str
    ec2_metadata: Union[EC2Metadata, None]
    run_elapsed_time_s: float
    number_of_jobs: int


class CollatedResults(TypedDict):
    metadata: Metadata
    results: List[Any]


class ResultCollatingCallback(Callback):
    def __init__(self) -> None:
        self._multirun_dir: Optional[Path] = None
        self._begin = 0
        self._end = 0

    def on_multirun_start(self, config: DictConfig, **kwargs: Any) -> None:
        self._begin = perf_counter()

    def on_job_start(self, config: DictConfig, **kwargs: Any) -> None:
        # Runtime variables like the output directory are not available in `on_multirun_end` is called, but are
        # available in `on_job_start`, so we collect the path here and refer to it later.
        if not self._multirun_dir:
            # should be something like "./multirun/2024-11-08/15-47-08/"
            self._multirun_dir = Path(config.hydra.runtime.output_dir).parent

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        self._end = perf_counter()
        run_elapsed_time = self._end - self._begin

        collated_results = self._collate_results(config, run_elapsed_time)
        collated_results_path = self._multirun_dir / _COLLATED_RESULTS_FILENAME

        logger.info("Saving collated results to: %s", collated_results_path)
        with open(collated_results_path, "w") as f:
            json.dump(collated_results, f, ensure_ascii=False, indent=4)
        logger.info("Collated results saved successfully")

    def _collate_results(
        self, config: DictConfig, run_elapsed_time: float
    ) -> CollatedResults:
        collated_results = []
        for file in self._multirun_dir.glob("*/**/result*.json"):
            collated_results.append(json.loads(file.read_text()))

        logger.info("Collated %i result files", len(collated_results))
        return {
            "metadata": {
                "python_version": sys.version,
                "pytorch_version": torch.__version__,
                "hydra_version": config.hydra.runtime.version,
                "ec2_metadata": get_ec2_metadata(),
                "run_elapsed_time_s": run_elapsed_time,
                "number_of_jobs": len(collated_results),
            },
            "results": collated_results,
        }


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
