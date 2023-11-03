"""
s3mapstyle_dataset.py
    API for accessing as PyTorch MapStyleDataset files stored in S3.
"""
from typing import List, Any

import torch.utils.data

from s3dataset_s3_client import S3Object
from . import S3DatasetBase

"""
s3mapstyle_dataset.py
    API for accessing as PyTorch Dataset files stored in S3. 
"""


class S3MapStyleDataset(S3DatasetBase, torch.utils.data.Dataset):
    @property
    def dataset_objects(self) -> List[S3Object]:
        if not isinstance(self._dataset_objects, list):
            self._dataset_objects = list(self._dataset_objects)
        return self._dataset_objects

    def __getitem__(self, i: int) -> Any:
        return self._transform(self.dataset_objects[i])

    def __len__(self):
        return len(self.dataset_objects)
