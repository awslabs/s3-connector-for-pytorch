#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class S3ClientConfig:
    """A dataclass exposing configurable parameters for the S3 client.

    Args:
    throughput_target_gbps(float): Throughput target in Gigabits per second (Gbps) that we are trying to reach.
        10.0 Gbps by default (may change in future).
    part_size(int): Size (bytes) of file parts that will be uploaded/downloaded.
        Note: for saving checkpoints, the inner client will adjust the part size to meet the service limits.
        (max number of parts per upload is 10,000, minimum upload part size is 5 MiB).
        Part size must have values between 5MiB and 5GiB.
        8MiB by default (may change in future).
    unsigned(bool): Set to true to disable signing S3 requests.
    force_path_style(bool): forceful path style addressing for S3 client.
    max_attempts(int): amount of retry attempts for retrieable errors.
    profile(str): Profile name to use for S3 authentication.
    """

    throughput_target_gbps: float = 10.0
    part_size: int = 8 * 1024 * 1024
    unsigned: bool = False
    force_path_style: bool = False
    max_attempts: int = 10
    profile: Optional[str] = None
