from typing import Iterator, Any

import torch

from . import S3DatasetBase

"""
s3iterable_dataset.py
    API for accessing as PyTorch IterableDataset files stored in S3. 
"""


class S3IterableDataset(S3DatasetBase, torch.utils.data.IterableDataset):
    def __iter__(self) -> Iterator[Any]:
        return map(self._transform, self._get_dataset_objects(self._get_client()))
