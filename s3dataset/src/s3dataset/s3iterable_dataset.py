from typing import Iterable, Iterator, Any, Callable

import torch
from s3dataset_s3_client._s3dataset import MountpointS3Client
from torch.utils.data import get_worker_info

from s3dataset_s3_client import S3Object
from . import S3DatasetBase
from .s3dataset_base import _identity

"""
s3iterable_dataset.py
    API for accessing as PyTorch IterableDataset files stored in S3. 
"""


class S3IterableDataset(S3DatasetBase, torch.utils.data.IterableDataset):
    def __init__(
        self,
        client: MountpointS3Client,
        dataset_objects: Iterable[S3Object] = (),
        transform: Callable[[S3Object], Any] = _identity,
    ):
        super().__init__(client, dataset_objects, transform)
        self._num_workers = 0
        self._worker_id = 0

    @property
    def dataset_objects(self) -> Iterable[S3Object]:
        return iter(self._dataset_objects)

    def __iter__(self) -> Iterator[Any]:
        return map(
            self._transform,
            map(
                _remove_enumerate,
                filter(self._filter_func, enumerate(self._dataset_objects)),
            ),
        )

    @staticmethod
    def worker_init(worker_id: int):
        worker_info = get_worker_info()
        worker_info.dataset._num_workers = worker_info.num_workers
        worker_info.dataset._worker_id = worker_info.id

    def _filter_func(self, enumerate_result):
        # When there are no workers, self._num_workers is unchanged from __init__ and is 0.
        if self._num_workers == 0:
            return True
        # By default, PyTorch runs every worker on every item of the dataset.
        # This function 'groups' items by worker id like so:
        # [(0, obj0), (1, obj1), ..., (9, obj9)], _num_workers = 3
        # w0 -> [0, ..., 9] -> i % 3 == 0 -> [0, 3, 6, 9]
        # w1 -> [0, ..., 9] -> i % 3 == 1 -> [1, 4, 7]
        # w2 -> [0, ..., 9] -> i % 3 == 2 -> [2, 5, 8]
        return enumerate_result[0] % self._num_workers == self._worker_id


def _remove_enumerate(enumerate_result):
    return enumerate_result[1]
