from typing import Iterable, Iterator, Any, Callable

import torch
from s3dataset_s3_client._s3dataset import MountpointS3Client, S3DatasetException
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
            _FilterUsingId(self._dataset_objects, self._filter_func),
        )

    @staticmethod
    def worker_init(worker_id: int):
        worker_info = get_worker_info()
        if worker_info is None:
            raise S3DatasetException("worker_init must only be called from a worker")
        if not isinstance(worker_info.dataset, S3IterableDataset):
            raise S3DatasetException(
                "S3IterableDataset.worker_init can only be used on S3IterableDataset datasets."
            )
        worker_info.dataset._num_workers = worker_info.num_workers
        worker_info.dataset._worker_id = worker_info.id

    def _filter_func(self, current_id: int) -> bool:
        # When there are no workers, self._num_workers is unchanged from __init__ and is 0.
        if self._num_workers == 0:
            return True
        # By default, PyTorch runs every worker on every item of the dataset.
        # This function 'groups' items by worker id like so:
        # [(0, obj0), (1, obj1), ..., (9, obj9)], _num_workers = 3
        # w0 -> [0, ..., 9] -> i % 3 == 0 -> [0, 3, 6, 9]
        # w1 -> [0, ..., 9] -> i % 3 == 1 -> [1, 4, 7]
        # w2 -> [0, ..., 9] -> i % 3 == 2 -> [2, 5, 8]
        return current_id % self._num_workers == self._worker_id


class _FilterUsingId:
    def __init__(self, iterable: Iterable, filter_function: Callable[[int], bool]):
        self.filter_function = filter_function
        self.iterator = enumerate(iterable)

    def __iter__(self):
        return self

    def __next__(self):
        i, data = next(self.iterator)
        while not self.filter_function(i):
            i, data = next(self.iterator)
        return data
