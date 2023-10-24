import torch

from s3dataset.s3dataset_base import S3DatasetBase

"""
s3iterable_dataset.py
    API for accessing as PyTorch IterableDataset files stored in S3 . 
"""


class S3IterableDataset(S3DatasetBase, torch.utils.data.IterableDataset):
    def __iter__(self):
        return self.dataset_objects
