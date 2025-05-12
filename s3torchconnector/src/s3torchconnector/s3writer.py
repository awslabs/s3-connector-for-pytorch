#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import io
from typing import Union
import threading
import logging

from s3torchconnectorclient._mountpoint_s3_client import PutObjectStream

logger = logging.getLogger(__name__)


class S3Writer(io.BufferedIOBase):
    """A write-only, file like representation of a single object stored in S3."""

    def __init__(self, stream: PutObjectStream):
        self.stream = stream
        self._position = 0
        self._closed = False
        self._lock = threading.Lock()

    def __enter__(self):
        self._position = 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close stream on normal exit, log any exceptions that occurred."""
        if exc_type is not None:
            try:
                logger.info(
                    f"Exception occurred before closing stream: {exc_type.__name__}: {exc_val}"
                )
            except:
                pass
        else:
            self.close()

    def write(
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
        self._position += len(data)
        return len(data)

    def close(self):
        """Close write-stream to S3. Ensures all bytes are written successfully.

        Raises:
            S3Exception: An error occurred accessing S3.
        """
        with self._lock:
            if not self._closed:
                self._closed = True
                self.stream.close()

    def flush(self):
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

    def tell(self) -> int:
        """
        Returns:
              int: Current stream position.
        """
        return self._position
