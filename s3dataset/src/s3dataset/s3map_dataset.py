#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from typing import List, Any, Callable, Iterable

import torch.utils.data

from ._s3client import S3Client, S3Reader

from . import S3DatasetBase
from .s3dataset_base import _identity

"""
s3map_dataset.py
    A Map-Style Dataset for objects stored in S3. 
"""


class S3MapDataset(S3DatasetBase, torch.utils.data.Dataset):
    def __init__(
        self,
        region: str,
        get_dataset_objects: Callable[[S3Client], Iterable[S3Reader]],
        transform: Callable[[S3Reader], Any] = _identity,
    ):
        super().__init__(region, get_dataset_objects, transform)
        self._dataset_object_store = None

    @property
    def _dataset_objects(self) -> List[S3Reader]:
        if self._dataset_object_store is None:
            self._dataset_object_store = list(
                self._get_dataset_objects(self._get_client())
            )
        return self._dataset_object_store

    def __getitem__(self, i: int) -> Any:
        return self._transform(self._dataset_objects[i])

    def __len__(self):
        return len(self._dataset_objects)
