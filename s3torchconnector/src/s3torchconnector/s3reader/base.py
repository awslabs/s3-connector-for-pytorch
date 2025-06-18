#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import io
from io import SEEK_SET
from abc import ABC, abstractmethod
from typing import Optional

class _BaseS3Reader(ABC, io.BufferedIOBase):
    """Abstract base class for S3 reader implementations."""

    @property
    @abstractmethod
    def bucket(self) -> str:
        pass

    @property
    @abstractmethod
    def key(self) -> str:
        pass

    @abstractmethod
    def read(self, size: Optional[int] = None) -> bytes:
        pass

    @abstractmethod
    def seek(self, offset: int, whence: int = SEEK_SET, /) -> int:
        pass

    @abstractmethod
    def tell(self) -> int:
        pass

    @abstractmethod
    def readinto(self, buf) -> int:
        pass