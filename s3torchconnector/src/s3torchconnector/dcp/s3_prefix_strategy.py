#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
from abc import ABC, abstractmethod
from typing import List, Optional

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

    def __init__(
        self,
        base: int,
        epoch_num: Optional[int] = None,
        min_prefix_length: int = 10,
        prefix_count: Optional[int] = None,
    ):
        """
            Initialize numeric prefix strategy.

        Args:
            base (int): The numeric base for the prefix (e.g., 2 for binary, 16 for hex).
            epoch_num (int, optional): Epoch number for checkpoint ordering. If None,
                no epoch information will be included in the prefix. Defaults to None.
            min_prefix_length (int): Minimum length of the generated prefix. Prefix will be
                padded with trailing zeros if necessary. Must be positive. Defaults to 10.
            prefix_count (int, optional): Number of unique prefixes to generate. If not provided,
                world size will be used as default value. Defaults to None.

        Raises:
            ValueError: If epoch_num, min_prefix_length, or prefix_count are invalid.
        """
        if min_prefix_length < 1:
            raise ValueError(
                f"Minimum prefix length must be positive, got {min_prefix_length}"
            )

        if epoch_num is not None and not isinstance(epoch_num, int):
            raise ValueError(
                f"Epoch number must be None or an integer, got {epoch_num}"
            )

        if prefix_count is not None and (
            not isinstance(prefix_count, int) or prefix_count < 1
        ):
            raise ValueError(
                f"Prefix count must be a positive integer, got {prefix_count}"
            )

        super().__init__()
        self.base = base
        self.epoch_num = epoch_num
        self.min_prefix_len = min_prefix_length

        self.prefix_count = 1
        if prefix_count is not None:
            self.prefix_count = prefix_count
        elif dist.is_initialized():
            self.prefix_count = dist.get_world_size()

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
        minimum_required_length = self._calculate_prefix_length()
        adjusted_prefix_length = max(minimum_required_length, self.min_prefix_len)

        all_prefixes = [
            self._format_number(i, adjusted_prefix_length)[::-1]
            for i in range(self.prefix_count)
        ]

        return all_prefixes

    def _calculate_prefix_length(self) -> int:
        """Calculate minimum prefix length needed for unique combinations."""
        prefix_length = 1
        size = self.base
        while size < self.prefix_count:
            prefix_length += 1
            size *= self.base
        return prefix_length

    @abstractmethod
    def _format_number(self, number: int, length: int) -> str:
        """Format a number to the appropriate base representation."""
        pass


class BinaryPrefixStrategy(NumericPrefixStrategy):
    """Binary (Base2) prefix generation strategy using only 0 and 1."""

    def __init__(
        self,
        epoch_num: Optional[int] = None,
        min_prefix_length: int = 10,
        prefix_count: Optional[int] = None,
    ):
        super().__init__(
            base=2,
            epoch_num=epoch_num,
            min_prefix_length=min_prefix_length,
            prefix_count=prefix_count,
        )

    def _format_number(self, number: int, length: int) -> str:
        return format(number, f"0{length}b")


class HexPrefixStrategy(NumericPrefixStrategy):
    """Hexadecimal-based prefix generation strategy."""

    def __init__(
        self,
        epoch_num: Optional[int] = None,
        min_prefix_length: int = 10,
        prefix_count: Optional[int] = None,
    ):
        super().__init__(
            base=16,
            epoch_num=epoch_num,
            min_prefix_length=min_prefix_length,
            prefix_count=prefix_count,
        )

    def _format_number(self, number: int, length: int) -> str:
        return format(number, f"0{length}x")


class RoundRobinPrefixStrategy(S3PrefixStrategyBase):
    """Strategy that distributes ranks across user-provided prefixes in round-robin fashion."""

    def __init__(self, user_prefixes: List[str], epoch_num: Optional[int] = None):
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
