#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from typing import Any, Union, Optional

import boto3
import requests
import torch
from botocore.exceptions import ClientError
from hydra.experimental.callback import Callback
from omegaconf import DictConfig

import s3torchconnector
from s3torchbenchmarking.constants import (
    JOB_RESULTS_FILENAME,
    RUN_FILENAME,
    Run,
    EC2Metadata,
    URL_IMDS_TOKEN,
    URL_IMDS_DOCUMENT,
)

logger = logging.getLogger(__name__)


class ResultCollatingCallback(Callback):
    """Hydra callback (https://hydra.cc/docs/experimental/callbacks/).

    Defines some routines to execute when a benchmark run is finished: namely, to merge all job results
    ("job_results.json" files) in one place ("run.json" file), augmented with some metadata.
    """

    def __init__(self) -> None:
        self._multirun_path: Optional[Path] = None
        self._begin = 0

    def on_multirun_start(self, config: DictConfig, **kwargs: Any) -> None:
        self._begin = perf_counter()

    def on_job_start(self, config: DictConfig, **kwargs: Any) -> None:
        if not self._multirun_path:
            # Hydra variables like `hydra.runtime.output_dir` are not available inside :func:`on_multirun_end`, so we
            # get the information here.
            self._multirun_path = Path(config.hydra.runtime.output_dir).parent

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        run_elapsed_time = perf_counter() - self._begin
        run = self._build_run(config, run_elapsed_time)

        self._save_to_disk(run)
        if "dynamodb" in config:
            self._write_to_dynamodb(config.dynamodb.region, config.dynamodb.table, run)
        else:
            logger.info("DynamoDB config not provided: skipping write to table...")

    def _build_run(self, config: DictConfig, run_elapsed_time: float) -> Run:
        all_job_results = []
        for entry in self._multirun_path.glob(f"**/{JOB_RESULTS_FILENAME}"):
            if entry.is_file():
                all_job_results.append(json.loads(entry.read_text()))

        logger.info("Collected %i job results", len(all_job_results))
        return {
            "run_id": str(uuid.uuid4()),
            "timestamp_utc": datetime.now(timezone.utc).timestamp(),
            "scenario": config.hydra.job.name,
            "versions": {
                "python": sys.version,
                "pytorch": torch.__version__,
                "hydra": config.hydra.runtime.version,
                "s3torchconnector": s3torchconnector.__version__,
            },
            "ec2_metadata": _get_ec2_metadata(),
            "run_elapsed_time_s": run_elapsed_time,
            "number_of_jobs": len(all_job_results),
            "all_job_results": all_job_results,
        }

    def _save_to_disk(self, run: Run) -> None:
        run_filepath = self._multirun_path / RUN_FILENAME

        logger.info("Saving run to: %s", run_filepath)
        with open(run_filepath, "w") as f:
            json.dump(run, f, ensure_ascii=False, indent=2)
        logger.info("Run saved successfully")

    @staticmethod
    def _write_to_dynamodb(region: str, table_name: str, run: Run) -> None:
        dynamodb = boto3.resource("dynamodb", region_name=region)
        table = dynamodb.Table(table_name)

        # `parse_float=Decimal` is required for DynamoDB (the latter does not work with floats), so we perform that
        # (strange) conversion through dumping then loading again the :class:`Run` object.
        run_json = json.loads(json.dumps(run), parse_float=Decimal)

        try:
            logger.info("Putting item into table: %s", table_name)
            table.put_item(Item=run_json)
            logger.info("Put item into table successfully")
        except ClientError:
            logger.error("Couldn't put item into table %s", table, exc_info=True)


def _get_ec2_metadata() -> Union[EC2Metadata, None]:
    """Get some EC2 metadata.

    See also https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instancedata-data-retrieval.html#instancedata-inside-access.
    """
    token = requests.put(
        URL_IMDS_TOKEN,
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
        timeout=5.0,
    )
    if token.status_code != 200:
        logger.warning("Failed to get EC2 metadata (acquiring token): %s", token)
        return None

    document = requests.get(
        URL_IMDS_DOCUMENT, headers={"X-aws-ec2-metadata-token": token.text}, timeout=5.0
    )
    if document.status_code != 200:
        logger.warning("Failed to get EC2 metadata (fetching document): %s", document)
        return None

    payload = document.json()
    return {
        "architecture": payload["architecture"],
        "image_id": payload["imageId"],
        "instance_type": payload["instanceType"],
        "region": payload["region"],
    }
