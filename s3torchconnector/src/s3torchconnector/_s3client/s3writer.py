#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import io
from typing import Union

from s3torchconnectorclient._mountpoint_s3_client import PutObjectStream

"""
s3writer.py
    File like representation of a writeable S3 object.
"""


class S3Writer(io.BufferedIOBase):
    def __init__(self, stream: PutObjectStream):
        self.stream = stream

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def write(self, data: Union[bytes, memoryview]) -> int:
        if isinstance(data, memoryview):
            data = data.tobytes()
        self.stream.write(data)
        return len(data)

    def close(self):
        self.stream.close()

    def flush(self):
        pass

    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True
