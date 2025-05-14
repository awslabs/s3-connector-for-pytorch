#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import pytest
from unittest.mock import patch
from s3torchconnector.dcp.s3_prefix_strategy import (
    S3PrefixStrategyBase,
    DefaultPrefixStrategy,
    BinaryPrefixStrategy,
    RoundRobinPrefixStrategy,
    HexPrefixStrategy,
)


def test_default_strategy_generate_prefix():
    """Test the generate_prefix method of DefaultPrefixStrategy."""
    default_strategy = DefaultPrefixStrategy()
    test_cases = [(0, "__0_"), (1, "__1_"), (100, "__100_"), (-1, "__-1_")]

    for rank, expected in test_cases:
        result = default_strategy.generate_prefix(rank)
        assert result == expected


def test_call_method():
    """Test the __call__ method of the strategy."""
    rank = 5
    expected = "__5_"
    default_strategy = DefaultPrefixStrategy()
    result = default_strategy(rank)
    assert result == expected


def test_base_class_is_abstract():
    """Test that S3PrefixStrategyBase cannot be instantiated directly."""
    with pytest.raises(TypeError):
        S3PrefixStrategyBase()


def test_custom_strategy():
    """Test creating and using a custom strategy."""

    class CustomPrefixStrategy(S3PrefixStrategyBase):
        def generate_prefix(self, rank: int) -> str:
            return f"rank_{rank}/data_"

    custom_strategy = CustomPrefixStrategy()
    assert custom_strategy.generate_prefix(1) == "rank_1/data_"
    assert custom_strategy(2) == "rank_2/data_"


@patch("torch.distributed.get_world_size")
@patch("torch.distributed.is_initialized")
def test_hex_prefix_strategy(is_initialized_mock, get_world_size_mock):
    """Test the HexPrefixStrategy."""
    world_size = 16
    is_initialized_mock.return_value = True
    get_world_size_mock.return_value = world_size
    hex_strategy = HexPrefixStrategy()
    assert hex_strategy.generate_prefix(10) == "a000000000/__10_"
    assert hex_strategy(255) == "f000000000/__255_"

    hex_strategy = HexPrefixStrategy(epoch_num=5)
    assert hex_strategy.generate_prefix(10) == "a000000000/epoch_5/__10_"
    assert hex_strategy(255) == "f000000000/epoch_5/__255_"


@patch("torch.distributed.get_world_size")
@patch("torch.distributed.is_initialized")
def test_binary_prefix_strategy(is_initialized_mock, get_world_size_mock):
    """Test the BinaryPrefixStrategy."""
    # Test without distributed initialization
    is_initialized_mock.return_value = False
    binary_strategy = BinaryPrefixStrategy()
    assert binary_strategy.generate_prefix(0) == "0000000000/__0_"
    assert binary_strategy(1) == "0000000000/__1_"

    # Test with distributed initialization
    is_initialized_mock.return_value = True
    get_world_size_mock.return_value = 4
    binary_strategy = BinaryPrefixStrategy()
    assert binary_strategy.generate_prefix(0) == "0000000000/__0_"
    assert binary_strategy.generate_prefix(3) == "1100000000/__3_"
    assert binary_strategy.prefix_map == [
        "0000000000",
        "1000000000",
        "0100000000",
        "1100000000",
    ]

    # Test with epoch number
    binary_strategy = BinaryPrefixStrategy(epoch_num=3)
    assert binary_strategy.generate_prefix(0) == "0000000000/epoch_3/__0_"
    assert binary_strategy(3) == "1100000000/epoch_3/__3_"


def test_round_robin_prefix_strategy():
    """Test the RoundRobinPrefixStrategy."""
    # Test basic functionality
    prefixes = ["prefix1", "prefix2", "prefix3"]
    rr_strategy = RoundRobinPrefixStrategy(prefixes)

    assert rr_strategy.generate_prefix(0) == "prefix1/__0_"
    assert rr_strategy.generate_prefix(1) == "prefix2/__1_"
    assert rr_strategy.generate_prefix(2) == "prefix3/__2_"
    assert rr_strategy.generate_prefix(3) == "prefix1/__3_"
    assert rr_strategy.user_prefixes == prefixes

    # Test with epoch number
    rr_strategy = RoundRobinPrefixStrategy(prefixes, epoch_num=5)
    assert rr_strategy.generate_prefix(0) == "prefix1/epoch_5/__0_"
    assert rr_strategy(4) == "prefix2/epoch_5/__4_"

    # Test empty prefixes
    with pytest.raises(ValueError):
        RoundRobinPrefixStrategy([])

    # Test single prefix
    single_prefix_strategy = RoundRobinPrefixStrategy(["prefix"])
    assert single_prefix_strategy.generate_prefix(0) == "prefix/__0_"
    assert single_prefix_strategy.generate_prefix(1) == "prefix/__1_"


@patch("torch.distributed.get_world_size")
@patch("torch.distributed.is_initialized")
def test_hex_prefix_strategy_extended(is_initialized_mock, get_world_size_mock):
    """Additional tests for HexPrefixStrategy."""
    # Test large world size
    is_initialized_mock.return_value = True
    get_world_size_mock.return_value = 257  # Requires 3 hex digits

    hex_strategy = HexPrefixStrategy()
    assert hex_strategy.generate_prefix(256) == "0010000000/__256_"
    assert hex_strategy(257) == "0000000000/__257_"
    assert hex_strategy(0) == "0000000000/__0_"
    assert hex_strategy(100) == "4600000000/__100_"
    assert len(hex_strategy.prefix_map) == 257

    # Test small world size
    get_world_size_mock.return_value = 2
    hex_strategy = HexPrefixStrategy()
    assert hex_strategy.generate_prefix(0) == "0000000000/__0_"
    assert hex_strategy.generate_prefix(1) == "1000000000/__1_"
