#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import platform
import torch
from s3torchconnector import S3Reader
import boto3

from typing import Tuple, List


def _get_fork_methods() -> List[str]:
    """Get a set of valid start methods for PyTorch's multiprocessing.
    On macOS, the 'fork' and 'forkserver' start methods are known to crash,
    despite being reported as usable by PyTorch. This function filters out
    those methods for macOS systems.

    Returns:
        List[str]: A set of valid start methods for the current platform.
    """
    methods = set(torch.multiprocessing.get_all_start_methods())

    if platform.system() == "Darwin":
        # fork and forkserver crash on MacOS, even though it's reported as usable.
        # https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
        # https://bugs.python.org/issue?@action=redirect&bpo=33725
        methods -= {"fork", "forkserver"}
    return [method for method in methods]


def _set_start_method(start_method: str):
    torch.multiprocessing.set_start_method(start_method, force=True)


def _read_data(s3reader: S3Reader) -> Tuple[str, bytes]:
    return s3reader.key, s3reader.read()


def _list_folders_in_bucket(bucket_name, prefix=""):
    if prefix and not prefix.endswith('/'):
        prefix += '/'

    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')

    pages = paginator.paginate(
        Bucket=bucket_name,
        Delimiter='/',
        Prefix=prefix
    )

    folders = []
    for page in pages:
        # Common prefixes are the folders
        if 'CommonPrefixes' in page:
            for obj in page['CommonPrefixes']:
                folder_name = obj['Prefix']
                if prefix:
                    # Remove the prefix from the folder name if it exists
                    folder_name = folder_name[len(prefix):]
                if folder_name:  # Avoid empty folder names
                    folders.append(folder_name.rstrip('/'))
    return folders

