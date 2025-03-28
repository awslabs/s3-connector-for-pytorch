#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
from hypothesis import given, example
from hypothesis.strategies import integers, floats

from s3torchconnector import S3ClientConfig
from .test_s3_client import MiB, GiB


def test_default():
    config = S3ClientConfig()
    assert config.part_size == 8 * MiB
    assert config.throughput_target_gbps == 10.0
    assert config.force_path_style is False
    assert config.max_attempts == 3


def test_enable_force_path_style():
    config = S3ClientConfig(force_path_style=True)
    assert config.force_path_style is True


@given(part_size=integers(min_value=5 * MiB, max_value=5 * GiB))
def test_part_size_setup(part_size: int):
    config = S3ClientConfig(part_size=part_size)
    assert config.part_size == part_size
    assert config.throughput_target_gbps == 10.0


@given(throughput_target_gbps=floats(min_value=1.0, max_value=100.0))
def test_throughput_target_gbps_setup(throughput_target_gbps: float):
    config = S3ClientConfig(throughput_target_gbps=throughput_target_gbps)
    assert config.part_size == 8 * 1024 * 1024
    assert config.throughput_target_gbps == throughput_target_gbps

@given(max_attempts=integers(min_value=1, max_value=10))
def test_max_attempts_setup(max_attempts: int):
    config = S3ClientConfig(max_attempts=max_attempts)
    assert config.max_attempts == max_attempts

@given(
    part_size=integers(min_value=5 * MiB, max_value=5 * GiB),
    throughput_target_gbps=floats(min_value=1.0, max_value=100.0),
    max_attempts=integers(min_value=1, max_value=10)
)
@example(part_size=5 * MiB, throughput_target_gbps=10.0,  max_attempts=2)
@example(part_size=5 * GiB, throughput_target_gbps=15.0,  max_attempts=8)
def test_custom_setup(part_size: int, throughput_target_gbps: float, max_attempts: int):
    config = S3ClientConfig(
        part_size=part_size, throughput_target_gbps=throughput_target_gbps, max_attempts=max_attempts
    )
    assert config.part_size == part_size
    assert config.throughput_target_gbps == throughput_target_gbps
    assert config.max_attempts == max_attempts
