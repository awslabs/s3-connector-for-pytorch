#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

"""Collect and write collated results to DynamoDB table.

This script collects all "collated_results.json" files in a given directory, and write them (in batch) to the specified
DynamoDB table.

Requires AWS credentials to be correctly set beforehand (env var).
"""

import argparse
import json
import logging
from decimal import Decimal
from pathlib import Path
from typing import List, Any

import boto3
from botocore.exceptions import ClientError

_RUN_FILENAME = "run.json"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def collect_collated_results(results_dir: str) -> List[Any]:
    all_job_results = []
    for entry in Path(results_dir).glob(f"**/{_RUN_FILENAME}"):
        if entry.is_file():
            with open(entry, "r") as f:
                all_job_results.append(json.load(f, parse_float=Decimal))

    logger.info("Collected %i job results", len(all_job_results))
    return all_job_results


def write_collated_results_to_dynamodb(
    collated_results: List[Any], region: str, table: str
) -> None:
    dynamodb = boto3.resource("dynamodb", region_name=region)

    try:
        with dynamodb.Table(table).batch_writer() as writer:
            for collated_result in collated_results:
                writer.put_item(Item=collated_result)
    except ClientError:
        logger.error("Couldn't load data into table %s", table, exc_info=True)


def main():
    parser = argparse.ArgumentParser(
        description="Collect and write collated results to DynamoDB table"
    )
    parser.add_argument(
        "collated_results_dir", help="directory where the collated results are"
    )
    parser.add_argument("region", help="region where the DynamoDB table is")
    parser.add_argument("table", help="DynamoDB table name")
    args = parser.parse_args()

    collated_results = collect_collated_results(args.collated_results_dir)
    write_collated_results_to_dynamodb(collated_results, args.region, args.table)


if __name__ == "__main__":
    main()
