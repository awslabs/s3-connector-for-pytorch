#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
from hypothesis import given
from hypothesis.strategies import integers, floats

from s3torchconnector import S3ClientConfig


def test_default():
    config = S3ClientConfig()
    assert config is not None
    assert config.part_size == 8 * 1024 * 1024
    assert abs(config.throughput_target_gbps - 10.0) < 1e-9


@given(part_size=integers(min_value=1, max_value=1e12))
def test_part_size_setup(part_size: int):
    config = S3ClientConfig(part_size=part_size)
    assert config is not None
    assert config.part_size == part_size
    assert abs(config.throughput_target_gbps - 10.0) < 1e-9


@given(throughput_target_gbps=floats(min_value=1.0, max_value=100.0))
def test_throughput_target_gbps_setup(throughput_target_gbps: float):
    config = S3ClientConfig(throughput_target_gbps=throughput_target_gbps)
    assert config is not None
    assert config.part_size == 8 * 1024 * 1024
    assert abs(config.throughput_target_gbps - throughput_target_gbps) < 1e-9


@given(
    part_size=integers(min_value=1, max_value=1e12),
    throughput_target_gbps=floats(min_value=1.0, max_value=100.0),
)
def test_custom_setup(part_size: int, throughput_target_gbps: float):
    config = S3ClientConfig(
        part_size=part_size, throughput_target_gbps=throughput_target_gbps
    )
    assert config is not None
    assert config.part_size == part_size
    assert abs(config.throughput_target_gbps - throughput_target_gbps) < 1e-9
