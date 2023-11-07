from s3dataset_s3_client._s3dataset import MountpointS3Client
from torch import multiprocessing
from torch.utils.data import DataLoader, get_worker_info

from s3dataset import S3IterableDataset
from s3dataset_s3_client import S3Object

E2E_TEST_BUCKET = "dataset-it-bucket"
E2E_BUCKET_PREFIX = "e2e-tests/images-10/img"
E2E_TEST_REGION = "eu-west-2"


class Counter:
    # A simple counter that's synced between processes.
    def __init__(self):
        self._value = multiprocessing.Value("i", 0)

    def increment(self, n: int = 1, /):
        with self._value.get_lock():
            self._value.value += n

    @property
    def value(self):
        return self._value.value


def test_s3iterable_dataset_multiprocess():
    counter = Counter()
    client = MountpointS3Client(E2E_TEST_REGION)
    dataset = S3IterableDataset.from_bucket(
        E2E_TEST_BUCKET,
        prefix=E2E_BUCKET_PREFIX,
        client=client,
        transform=_verify_worker_info,
    )

    num_workers = 3
    num_epochs = 5
    num_images = 10

    dataloader = DataLoader(dataset, num_workers=num_workers)
    for epoch in range(num_epochs):
        for i, _ in enumerate(dataloader):
            counter.increment()
        assert (i + 1) == num_images * num_workers
    assert counter.value == num_workers * num_epochs * num_images


def test_s3iterable_dataset_multiprocess_skips_files():
    counter = Counter()
    client = MountpointS3Client(E2E_TEST_REGION)
    dataset = S3IterableDataset.from_bucket(
        E2E_TEST_BUCKET,
        prefix=E2E_BUCKET_PREFIX,
        client=client,
        transform=_verify_worker_info,
    )

    num_workers = 3
    num_epochs = 5
    num_images = 10

    dataloader = DataLoader(
        dataset, num_workers=num_workers, worker_init_fn=dataset.worker_init
    )
    for epoch in range(num_epochs):
        for i, _ in enumerate(dataloader):
            counter.increment()
    assert counter.value == num_epochs * num_images


def _verify_worker_info(s3_object: S3Object) -> int:
    assert s3_object._stream is None
    worker_info = get_worker_info()
    assert worker_info is not None
    return 0
