#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import json
import logging
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from typing import Any, Union, Optional, List

import boto3
import requests
import torch
from botocore.exceptions import ClientError
from hydra import TaskFunction
from hydra.core.utils import JobReturn
from hydra.experimental.callback import Callback
from omegaconf import DictConfig, OmegaConf

import s3torchconnector
from s3torchbenchmarking.constants import (
    JOB_RESULTS_FILENAME,
    RUN_RESULTS_FILENAME,
    RunResults,
    EC2Metadata,
    URL_IMDS_TOKEN,
    URL_IMDS_DOCUMENT,
    JobResults,
)

logger = logging.getLogger(__name__)


class ResultCollatingCallback(Callback):
    """Hydra callbacks (https://hydra.cc/docs/experimental/callbacks/).

    Execute callback functions at job start, job end, multirun start, and multirun end. Help to save job and multirun
    results to local file, and (optionally) write run results to DynamoDB.
    """

    def __init__(self) -> None:
        self._multirun_path: Optional[Path] = None
        self._begin: float = 0
        self._job_results: List[JobResults] = []

    def on_multirun_start(self, config: DictConfig, **kwargs: Any) -> None:
        self._begin = perf_counter()

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        run_elapsed_time = perf_counter() - self._begin
        run_results = self._build_run_result(config, run_elapsed_time)

        run_results_filepath = Path(self._multirun_path, RUN_RESULTS_FILENAME)
        _save_to_disk("run", run_results_filepath, run_results)

        if "dynamodb" in config:
            _write_to_dynamodb(
                config.dynamodb.region, config.dynamodb.table, run_results
            )
        else:
            logger.info("DynamoDB config not provided: skipping write to table...")

    def on_job_start(
        self, config: DictConfig, *, task_function: TaskFunction, **kwargs: Any
    ) -> None:
        if not self._multirun_path:
            # Hydra variables like `hydra.runtime.output_dir` are not available inside :func:`on_multirun_end`, so we
            # get the information here.
            self._multirun_path = Path(config.hydra.runtime.output_dir).parent

    def on_job_end(
        self, config: DictConfig, job_return: JobReturn, **kwargs: Any
    ) -> None:
        job_results: JobResults = {
            "config": OmegaConf.to_container(job_return.cfg),
            "metrics": job_return.return_value["metrics"],
        }
        self._job_results.append(job_results)

        job_results_path = Path(config.hydra.runtime.output_dir, JOB_RESULTS_FILENAME)
        _save_to_disk("job results", job_results_path, job_results)

    def _build_run_result(
        self, config: DictConfig, run_elapsed_time: float
    ) -> RunResults:
        return {
            "s3torchconnector_version": s3torchconnector.__version__,
            "timestamp_utc": datetime.now(timezone.utc).timestamp(),
            "scenario": config.hydra.job.config_name,
            "disambiguator": config.get("disambiguator", None),
            "run_elapsed_time_s": run_elapsed_time,
            "versions": {
                "python": sys.version,
                "pytorch": torch.__version__,
                "hydra": config.hydra.runtime.version,
            },
            "ec2_metadata": _get_ec2_metadata(),
            "job_results": self._job_results,
        }


def _get_ec2_metadata() -> Union[EC2Metadata, None]:
    """Get some EC2 metadata.

    Note:
        See also https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instancedata-data-retrieval.html.
    """
    response = requests.put(
        URL_IMDS_TOKEN,
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
        timeout=5.0,
    )
    if response.status_code != 200:
        logger.warning("Failed to get EC2 metadata (acquiring token): %s", response)
        return None

    response = requests.get(
        URL_IMDS_DOCUMENT,
        headers={"X-aws-ec2-metadata-token": response.text},
        timeout=5.0,
    )
    if response.status_code != 200:
        logger.warning("Failed to get EC2 metadata (fetching document): %s", response)
        return None

    payload = response.json()
    return {
        "architecture": payload["architecture"],
        "image_id": payload["imageId"],
        "instance_type": payload["instanceType"],
        "region": payload["region"],
    }


def _save_to_disk(type_: str, filepath: Optional[Path], obj: Any) -> None:
    logger.info("Saving %s to: %s", type_, filepath)
    with open(filepath, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    logger.info("✅ Saved %s successfully", type_)


def _write_to_dynamodb(region: str, table_name: str, run: RunResults) -> None:
    dynamodb = boto3.resource("dynamodb", region_name=region)
    table = dynamodb.Table(table_name)

    # `parse_float=Decimal` is required for DynamoDB (the latter does not work with floats), so we perform that
    # (strange) conversion through dumping then loading again the :class:`Run` object.
    run_json = json.loads(json.dumps(run), parse_float=Decimal)

    try:
        logger.info("Putting item into table: %s", table_name)
        table.put_item(Item=run_json)
        logger.info("✅ Put item into table successfully")
    except ClientError:
        logger.error("❌ Couldn't put item into table %s", table, exc_info=True)
