#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
from abc import ABC, abstractmethod
from typing import List

import torch.distributed as dist


class S3PrefixStrategyBase(ABC):
    """Base class for S3 prefix generation strategies."""

    def __init__(self):
        pass

    def __call__(self, rank: int) -> str:
        """Generate prefix for given rank."""
        return self.generate_prefix(rank)

    @abstractmethod
    def generate_prefix(self, rank: int) -> str:
        """Generate storage prefix for the given rank."""
        pass


class DefaultPrefixStrategy(S3PrefixStrategyBase):
    """Default strategy for generating S3 prefixes."""

    def generate_prefix(self, rank: int) -> str:
        """Generate simple rank-based name without prefix."""
        return f"__{rank}_"


class NumericPrefixStrategy(S3PrefixStrategyBase):
    """Base class for numeric prefix generation strategies."""

    def __init__(self, base: int, epoch_num: int = None):
        """
        Initialize numeric prefix strategy.

        Args:
            base: The numeric base for the prefix (e.g., 2 for binary, 16 for hex).
            epoch_num: Epoch number for checkpoint ordering. If None,
                       no epoch information will be included in the prefix.
        """
        super().__init__()
        self.base = base
        self.epoch_num = epoch_num
        self.prefix_map = self._generate_prefix_map()

    def generate_prefix(self, rank: int) -> str:
        """
        Generate numeric-based prefix with optional epoch number.

        Args:
            rank: Process rank in the distributed environment.

        Returns:
            Prefix string in format: <pattern>/epoch_<num>/__<rank>_
            or <pattern>/__<rank>_ if no epoch number is provided.
        """
        epoch_suffix = f"epoch_{self.epoch_num}/" if self.epoch_num is not None else ""
        return f"{self.prefix_map[rank % len(self.prefix_map)]}/{epoch_suffix}__{rank}_"

    def _generate_prefix_map(self) -> List[str]:
        """Generate mapping of ranks to numeric-based prefixes."""
        world_size = 1
        if  dist.is_initialized():
            world_size = dist.get_world_size()
        prefix_length = self._calculate_prefix_length(world_size)

        all_prefixes = [
            self._format_number(i, prefix_length)
            for i in range(self.base**prefix_length)
        ]

        return all_prefixes[:world_size]

    def _calculate_prefix_length(self, world_size: int) -> int:
        """Calculate minimum prefix length needed for unique combinations."""
        prefix_length = 1
        while self.base**prefix_length < world_size:
            prefix_length += 1
        return prefix_length

    @abstractmethod
    def _format_number(self, number: int, length: int) -> str:
        """Format a number to the appropriate base representation."""
        pass


class BinaryPrefixStrategy(NumericPrefixStrategy):
    """Binary (Base2) prefix generation strategy using only 0 and 1."""

    def __init__(self, epoch_num: int = None):
        super().__init__(base=2, epoch_num=epoch_num)

    def _format_number(self, number: int, length: int) -> str:
        return format(number, f"0{length}b")


class HexPrefixStrategy(NumericPrefixStrategy):
    """Hexadecimal-based prefix generation strategy."""

    def __init__(self, epoch_num: int = None):
        super().__init__(base=16, epoch_num=epoch_num)

    def _format_number(self, number: int, length: int) -> str:
        return format(number, f"0{length}x")


class RoundRobinPrefixStrategy(S3PrefixStrategyBase):
    """Strategy that distributes ranks across user-provided prefixes in round-robin fashion."""

    def __init__(self, user_prefixes: List[str], epoch_num: int = None):
        """
        Initialize round-robin prefix strategy.

        Args:
            user_prefixes: List of prefixes to distribute ranks across.
                          Must not be empty.
            epoch_num: Epoch number for checkpoint ordering.

        Raises:
            ValueError: If user_prefixes is empty.
        """
        super().__init__()
        if not user_prefixes:
            raise ValueError("user_prefixes must not be empty")

        self.user_prefixes = user_prefixes
        self.epoch_num = epoch_num

    def generate_prefix(self, rank: int) -> str:
        """
        Generate prefix for given rank using round-robin distribution.

        Args:
            rank: Process rank in the distributed environment.

        Returns:
            Prefix string in format: <user_prefix>/epoch_<num>/__<rank>_
            or <user_prefix>/__<rank>_ if no epoch number is provided.
        """
        epoch_suffix = f"epoch_{self.epoch_num}/" if self.epoch_num is not None else ""
        return f"{self.user_prefixes[rank % len(self.user_prefixes)]}/{epoch_suffix}__{rank}_"

