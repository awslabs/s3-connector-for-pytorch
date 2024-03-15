#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
from __future__ import annotations

import io
from types import TracebackType
from typing import Union, IO, Iterable, Optional

from s3torchconnectorclient._mountpoint_s3_client import PutObjectStream


class S3Writer(io.BufferedIOBase, IO[bytes]):
    """A write-only, file like representation of a single object stored in S3."""

    def __init__(self, stream: PutObjectStream):
        self.stream = stream

    def __enter__(self) -> S3Writer:
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()

    def write(  # type: ignore
        self,
        # Ignoring the type for this as we don't currently support the Buffer protocol
        data: Union[bytes, memoryview],  # type: ignore
    ) -> int:
        """Write bytes to S3 Object specified by bucket and key

        Args:
            data (bytes | memoryview): bytes to write

        Returns:
            int: Number of bytes written

        Raises:
            S3Exception: An error occurred accessing S3.
        """
        if isinstance(data, memoryview):
            data = data.tobytes()
        self.stream.write(data)
        return len(data)

    def writelines(  # type: ignore
        self,
        # Ignoring the type for this as we don't currently support the Buffer protocol
        data: Iterable[bytes | memoryview],  # type: ignore
    ) -> None:
        for line in data:
            self.write(line)

    def close(self) -> None:
        """Close write-stream to S3. Ensures all bytes are written successfully.

        Raises:
            S3Exception: An error occurred accessing S3.
        """
        self.stream.close()

    def flush(self) -> None:
        """No-op"""
        pass

    def readable(self) -> bool:
        """
        Returns:
            bool: Return whether object was opened for reading.
        """
        return False

    def writable(self) -> bool:
        """
        Returns:
            bool: Return whether object was opened for writing.
        """
        return True
