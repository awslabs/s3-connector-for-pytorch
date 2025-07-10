#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import io
from io import SEEK_SET
from abc import ABC, abstractmethod
from typing import Optional


class S3Reader(ABC, io.BufferedIOBase):
    """An abstract base class for read-only, file-like representation of a single object stored in S3.

    This class defines the interface for S3 readers. Concrete implementations (SequentialS3Reader or
    RangedS3Reader extend this class. S3ReaderConstructor creates partial functions of these
    implementations, which are then completed by S3Client with the remaining required parameters.
    """

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

    def readable(self) -> bool:
        """
        Returns:
            bool: Return whether object was opened for reading.
        """
        return True

    def writable(self) -> bool:
        """
        Returns:
            bool: Return whether object was opened for writing.
        """
        return False
