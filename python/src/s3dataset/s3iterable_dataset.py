from __future__ import annotations

import torch

from s3dataset.s3dataset_base import S3DatasetBase


class S3IterableDataset(S3DatasetBase, torch.utils.data.IterableDataset):
    def __iter__(self):
        return self.dataset_objects
