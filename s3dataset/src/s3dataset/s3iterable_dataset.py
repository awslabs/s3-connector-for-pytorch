from typing import Iterable, Iterator, Any

import torch

from s3dataset_s3_client import S3Object
from . import S3DatasetBase

"""
s3iterable_dataset.py
    API for accessing as PyTorch IterableDataset files stored in S3. 
"""


class S3IterableDataset(S3DatasetBase, torch.utils.data.IterableDataset):
    @property
    def dataset_objects(self) -> Iterable[S3Object]:
        return self._dataset_objects

    def __iter__(self) -> Iterator[Any]:
        return map(self._transform, self._dataset_objects)
