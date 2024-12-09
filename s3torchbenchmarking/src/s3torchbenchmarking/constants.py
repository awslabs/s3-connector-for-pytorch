#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from typing import TypedDict, Union, Any, List

JOB_RESULTS_FILENAME = "job_results.json"
RUN_FILENAME = "run.json"

# URLs for EC2 metadata retrieval (IMDSv2)
URL_IMDS_TOKEN = "http://169.254.169.254/latest/api/token"
URL_IMDS_DOCUMENT = "http://169.254.169.254/latest/dynamic/instance-identity/document"


class Versions(TypedDict):
    python: str
    pytorch: str
    hydra: str
    s3torchconnector: str


class EC2Metadata(TypedDict):
    architecture: str
    image_id: str
    instance_type: str
    region: str


class Run(TypedDict):
    """Information about a Hydra run.

    Also, a :class:`Run` object will be inserted as-is in DynamoDB."""

    run_id: str  # PK (Partition Key)
    timestamp_utc: float  # SK (Sort Key)
    scenario: str
    versions: Versions
    ec2_metadata: Union[EC2Metadata, None]
    run_elapsed_time_s: float
    number_of_jobs: int
    all_job_results: List[Any]
