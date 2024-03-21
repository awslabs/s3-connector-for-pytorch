#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
from dataclasses import dataclass


@dataclass(frozen=True)
class S3ClientConfig:
    """A dataclass exposing configurable parameters for the S3 client.

    Args:
    throughput_target_gbps(float): Throughput target in Gigabits per second (Gbps) that we are trying to reach.
        10.0 Gbps by default (may change in future).
    part_size(int): Size, in bytes, of parts that files will be downloaded or uploaded in.
        Note: for saving checkpoints, the inner client will adjust the part size to meet the service limits.
        (max number of parts per upload is 10,000, minimum upload part size is 5 MiB).
        Part size must have values between 5MiB and 5GiB.
        8MB by default (may change in future).
    """

    throughput_target_gbps: float = 10.0
    part_size: int = 8 * 1024 * 1024
