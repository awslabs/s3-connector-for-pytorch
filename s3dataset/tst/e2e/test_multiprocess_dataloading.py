import os
import re
from collections import Counter

from s3dataset_s3_client._s3dataset import MountpointS3Client
from torch.utils.data import DataLoader, get_worker_info

from s3dataset import S3IterableDataset
from s3dataset_s3_client import S3Object

E2E_TEST_BUCKET = "dataset-it-bucket"
E2E_BUCKET_PREFIX = "e2e-tests/images-10/img"
E2E_TEST_REGION = "eu-west-2"
LOCAL_DATASET_RELATIVE_PATH = "../resources/images-10/"
ABSOLUTE_PATH = os.path.dirname(__file__)

image_key_re = re.compile(r"img\d+\.jpg$")


def test_s3iterable_dataset_multiprocess():
    counter = 0
    client = MountpointS3Client(E2E_TEST_REGION)
    dataset = S3IterableDataset.from_bucket(
        E2E_TEST_BUCKET,
        prefix=E2E_BUCKET_PREFIX,
        client=client,
        transform=_verify_object,
    )

    num_workers = 3
    num_epochs = 5
    num_images = 10

    dataloader = DataLoader(dataset, num_workers=num_workers)
    for epoch in range(num_epochs):
        worker_count = Counter()
        for worker_id, _num_workers in dataloader:
            counter += 1
            worker_count[worker_id.item()] += 1
            assert _num_workers == num_workers
        assert len(worker_count) == num_workers
        assert all(times_found == num_images for times_found in worker_count.values())
        assert sum(worker_count.values()) == num_images * num_workers
    assert counter == num_workers * num_epochs * num_images


def test_s3iterable_dataset_multiprocess_skips_files():
    counter = 0
    client = MountpointS3Client(E2E_TEST_REGION)
    dataset = S3IterableDataset.from_bucket(
        E2E_TEST_BUCKET,
        prefix=E2E_BUCKET_PREFIX,
        client=client,
        transform=_verify_object,
    )

    num_workers = 3
    num_epochs = 5
    num_images = 10

    dataloader = DataLoader(
        dataset, num_workers=num_workers, worker_init_fn=dataset.worker_init
    )
    for epoch in range(num_epochs):
        worker_count = Counter()
        for worker_id, _num_workers in dataloader:
            counter += 1
            worker_count[worker_id.item()] += 1
        assert sum(worker_count.values()) == num_images
    assert counter == num_epochs * num_images


def _verify_object(s3_object: S3Object) -> (int, int):
    assert s3_object._stream is None

    image_key = image_key_re.search(s3_object.key).group(0)
    local_file = os.path.join(ABSOLUTE_PATH, LOCAL_DATASET_RELATIVE_PATH, image_key)
    with open(local_file, "rb") as local_image:
        assert local_image.read() == s3_object.read()

    return _get_worker_info()


def _get_worker_info() -> (int, int):
    worker_info = get_worker_info()
    assert worker_info is not None
    return worker_info.id, worker_info.num_workers
