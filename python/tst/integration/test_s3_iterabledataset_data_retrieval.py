import logging
import pickle

import pytest
from s3dataset._s3_iterabledataset import S3IterableDataset, S3DatasetSource
from s3dataset._s3dataset import MountpointS3Client

from expected_file_contents import HELLO_WORLD, PICKLE_NUM_ITERDATASET_100, \
    PICKLE_STR_ITER_DATASET_10, TORCH_NUM_ITERDATASET_100, TORCH_STR_ITERDATASET_10
from pytorch_iterable_dataset_generator import StringIterableDataset

logging.basicConfig(format="%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s")
logging.getLogger().setLevel(1)

log = logging.getLogger(__name__)

TEST_REGION="eu-west-2"
TEST_BUCKET="dataset-it-bucket"


@pytest.mark.parametrize(
    "uris, expected_key, expected_content",
    [
        (
            [f"s3://{TEST_BUCKET}/sample-files/hello_world.txt"],
            "sample-files/hello_world.txt",
            HELLO_WORLD
        ),
        (
            [f"s3://{TEST_BUCKET}/iterable-datasets/torch-num-iterdataset-100.pkl"],
            "iterable-datasets/torch-num-iterdataset-100.pkl",
            TORCH_NUM_ITERDATASET_100
        ),
        (
            [f"s3://{TEST_BUCKET}/iterable-datasets/pickle-num-iterdataset-100.pkl"],
            "iterable-datasets/pickle-num-iterdataset-100.pkl",
            PICKLE_NUM_ITERDATASET_100
        ),
        (
            [f"s3://{TEST_BUCKET}/iterable-datasets/torch-str-iterdataset-10.pkl"],
            "iterable-datasets/torch-str-iterdataset-10.pkl",
            TORCH_STR_ITERDATASET_10
        ),
        (
            [f"s3://{TEST_BUCKET}/iterable-datasets/pickle-str-iterdataset-10.pkl"],
            "iterable-datasets/pickle-str-iterdataset-10.pkl",
            PICKLE_STR_ITER_DATASET_10
        ),
    ],
)
def test_s3_iterabledataset_content(uris: [str], expected_key: str, expected_content: bytes):
    client = MountpointS3Client(TEST_REGION)
    source = S3DatasetSource.from_object_uris(client, uris)
    s3_iterabledataset = S3IterableDataset(client, source)

    for result in s3_iterabledataset:
        assert result.bucket == TEST_BUCKET
        assert result.key == expected_key
        for bytes in result:
            assert bytes == expected_content


@pytest.mark.parametrize(
    "uris, expected_key, expected_size, expected_content",
    [
        (
            [f"s3://{TEST_BUCKET}/iterable-datasets/pickle-str-iterdataset-10.pkl"],
            "iterable-datasets/pickle-str-iterdataset-10.pkl",
            10,
            ["string_" + str(i) for i in range(10)]
        ),
        (
            [f"s3://{TEST_BUCKET}/iterable-datasets/pickle-num-iterdataset-100.pkl"],
            "iterable-datasets/pickle-num-iterdataset-100.pkl",
            100,
            [i for i in range (100)]
        )
    ]
)
def test_string_iterabledataset_pickle_load(uris, expected_key, expected_size, expected_content):
    client = MountpointS3Client(TEST_REGION)
    source = S3DatasetSource.from_object_uris(client, uris)
    s3_iterabledataset = S3IterableDataset(client, source)

    for result in s3_iterabledataset:
        assert result.bucket == TEST_BUCKET
        assert result.key == expected_key
        unpickled_result = pickle.load(result)
        assert unpickled_result != None
        i:int = 0
        for str in unpickled_result:
            assert str == expected_content[i]
            i = i + 1

# TODO: Add Torch support
# Tried a similar test like the one above with torch.load
# io.UnsupportedOperation: seek. You can only torch.load from a file that is seekable.
# Please pre-load the data into a buffer like io.BytesIO and try to load from it instead.

def test_iterabledataset_from_bucket():
    client = MountpointS3Client(TEST_REGION)
    source = S3DatasetSource.from_bucket(client, TEST_BUCKET)
    s3_iterabledataset = S3IterableDataset(client, source)

    for result in s3_iterabledataset:
        assert result.bucket == TEST_BUCKET
        if (result.key.endswith("pickle-str-iterdataset-10.pkl")):
            assert result.object_info.size == 88
            string_iterable_dataset: StringIterableDataset = pickle.load(result)
            i: int = 0
            for str in string_iterable_dataset:
                assert str == f"string_{i}"
                i = i + 1
        elif (result.key.endswith("pickle-num-iterdataset-100.pkl")):
            assert result.object_info.size == 88
        elif (result.key.endswith("hello_world.txt")):
            assert result.object_info.size == 14

