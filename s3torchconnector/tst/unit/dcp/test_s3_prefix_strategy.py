#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import pytest
from unittest.mock import patch, MagicMock

from hypothesis import given

from s3torchconnector.dcp.s3_prefix_strategy import DefaultPrefixStrategy, S3PrefixStrategyBase
from s3torchconnector.dcp.s3_prefix_strategy import HexPrefixStrategy


def test_default_strategy_generate_prefix():
    """Test the generate_prefix method of DefaultPrefixStrategy."""
    default_strategy = DefaultPrefixStrategy()
    test_cases = [
        (0, "__0_"),
        (1, "__1_"),
        (100, "__100_"),
        (-1, "__-1_")
    ]

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
def test_hex_prefix_strategy(is_initialized_mock,
                             get_world_size_mock):
    """Test the HexPrefixStrategy."""
    world_size = 16
    is_initialized_mock.return_value = True
    get_world_size_mock.return_value = world_size
    hex_strategy = HexPrefixStrategy()
    assert hex_strategy.generate_prefix(10) == "0xa_"
    assert hex_strategy(255) == "0xff_"

    hex_strategy = HexPrefixStrategy(epoch_num=5)
    assert hex_strategy.generate_prefix(10) == "0xa_"
    assert hex_strategy(255) == "0xff_"


