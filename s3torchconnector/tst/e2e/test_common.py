#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import platform
import torch
from s3torchconnector import S3Reader

from typing import Tuple


def _get_fork_methods() -> set[str]:
    """Get a set of valid start methods for PyTorch's multiprocessing.
    On macOS, the 'fork' and 'forkserver' start methods are known to crash,
    despite being reported as usable by PyTorch. This function filters out
    those methods for macOS systems.

    Returns:
        set: A set of valid start methods for the current platform.
    """
    methods = set(torch.multiprocessing.get_all_start_methods())

    if platform.system() == "Darwin":
        # fork and forkserver crash on MacOS, even though it's reported as usable.
        # https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
        # https://bugs.python.org/issue?@action=redirect&bpo=33725
        methods -= {"fork", "forkserver"}
    return methods


def _set_start_method(start_method: str):
    torch.multiprocessing.set_start_method(start_method, force=True)


def _read_data(s3reader: S3Reader) -> Tuple[str, bytes]:
    return s3reader.key, s3reader.read()
