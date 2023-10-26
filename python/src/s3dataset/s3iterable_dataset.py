from typing import Iterable, Iterator, Any

import torch

from s3dataset.s3dataset_base import S3DatasetBase
from s3dataset.s3object import S3Object

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
