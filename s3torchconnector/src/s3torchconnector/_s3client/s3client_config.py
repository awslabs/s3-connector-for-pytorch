#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
from dataclasses import dataclass
from typing import Optional

# Upper bound for `max_attempts`, enforced Python-side to avoid an opaque
# pyo3_runtime.PanicException raised from inside the Rust client constructor
# (see issue #361). The AWS CRT retry strategy uses exponential backoff and
# rejects attempt counts above this limit. The exact value is a CRT-internal
# constant; 63 matches the value cited by the maintainer. It is centralized
# here so it can be adjusted in one place if the CRT limit is confirmed to
# differ.
_MAX_ATTEMPTS_LIMIT = 63


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

    def __post_init__(self) -> None:
        # Validate `max_attempts` here so that invalid values raise a clear,
        # catchable ValueError instead of an opaque pyo3_runtime.PanicException
        # from inside the Rust client constructor (see issue #361). The Rust
        # side calls NonZeroUsize::try_from(max_attempts).expect(...) (panics on
        # 0) and the AWS CRT retry strategy panics on values above its
        # exponential-backoff limit.
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        if self.max_attempts > _MAX_ATTEMPTS_LIMIT:
            raise ValueError(
                f"max_attempts must be between 1 and {_MAX_ATTEMPTS_LIMIT} "
                "(AWS CRT retry-strategy limit)"
            )
