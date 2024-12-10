#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from typing import TypedDict, Union, Any, List

JOB_RESULTS_FILENAME = "job_results.json"
RUN_RESULTS_FILENAME = "run_results.json"

# URLs for EC2 metadata retrieval (IMDSv2)
URL_IMDS_TOKEN = "http://169.254.169.254/latest/api/token"
URL_IMDS_DOCUMENT = "http://169.254.169.254/latest/dynamic/instance-identity/document"


class JobResults(TypedDict):
    """Results from a Hydra job."""

    config: Any
    metrics: Any


class Versions(TypedDict):
    """Version numbers (Python, PyTorch, and other libraries)."""

    python: str
    pytorch: str
    hydra: str


class EC2Metadata(TypedDict):
    """EC2 metadata (fetched from IMDSv2)."""

    architecture: str
    image_id: str
    instance_type: str
    region: str


class RunResults(TypedDict):
    """Results from a Hydra run.

    Note:
        An instance of :class:`RunResults` will be inserted as-is in DynamoDB.
    """

    s3torchconnector_version: str  # PK (Partition Key)
    timestamp_utc: float  # SK (Sort Key)
    scenario: str
    disambiguator: Union[str, None]  # helps to identify multi-instances benchmarks
    run_elapsed_time_s: float
    versions: Versions
    ec2_metadata: Union[EC2Metadata, None]
    job_results: List[JobResults]
