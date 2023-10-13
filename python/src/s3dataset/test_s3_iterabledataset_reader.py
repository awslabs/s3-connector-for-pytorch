import logging
import pickle

from s3dataset._s3dataset import MountpointS3Client

from s3dataset._s3_iterabledataset import S3IterableDataset, S3DatasetSource
from utils.pytorch_iterable_dataset_generator import NumberIterableDataset, StringIterableDataset

logging.basicConfig(format='%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s')
logging.getLogger().setLevel(1)

log = logging.getLogger(__name__)

def test_number_iterabledataset():
    client = MountpointS3Client('eu-west-2')
    uris = ['s3://dataset-it-bucket/iterable-datasets/num-iterdataset-100.pkl']
    source = S3DatasetSource.from_object_uris(client, uris)
    iterabledataset_reader = S3IterableDataset(client, source)

    for reader in iterabledataset_reader:
        assert reader.bucket == 'dataset-it-bucket'
        assert reader.key == 'iterable-datasets/num-iterdataset-100.pkl'
        number_iterable_dataset: NumberIterableDataset = pickle.load(reader)
        i: int = 0
        for number in number_iterable_dataset:
            assert number == i
            i = i + 1

def test_string_iterabledataset():
    client = MountpointS3Client('eu-west-2')
    uris = ['s3://dataset-it-bucket/iterable-datasets/str-iterdataset-10.pkl']
    source = S3DatasetSource.from_object_uris(client, uris)
    iterabledataset_reader = S3IterableDataset(client, source)

    for reader in iterabledataset_reader:
        assert reader.bucket == 'dataset-it-bucket'
        assert reader.key == 'iterable-datasets/str-iterdataset-10.pkl'
        string_iterable_dataset: StringIterableDataset = pickle.load(reader)
        i: int = 0
        for str in string_iterable_dataset:
            assert str == f"string_{i}"
            i = i + 1

def test_iterabledataset_reader():
    client = MountpointS3Client('eu-west-2')
    uris = ['s3://dataset-it-bucket/iterable-datasets/num-iterdataset-100.pkl', 's3://dataset-it-bucket/iterable-datasets/str-iterdataset-10.pkl']
    source = S3DatasetSource.from_object_uris(client, uris)
    iterabledataset_reader = S3IterableDataset(client, source)

    for reader in iterabledataset_reader:
        assert reader.bucket == 'dataset-it-bucket'
        assert reader.key.startswith('iterable-datasets')
        assert reader.key.endswith('.pkl')

def test_iterabledataset_from_bucket():
    client = MountpointS3Client('eu-west-2')
    source = S3DatasetSource.from_bucket(client, 'dataset-it-bucket')
    iterabledataset_reader = S3IterableDataset(client, source)

    for reader in iterabledataset_reader:
        assert reader.bucket == 'dataset-it-bucket'
        if (reader.key.endswith('str-iterdataset-10.pkl')):
            assert reader.object_info.size == 62
            string_iterable_dataset: StringIterableDataset = pickle.load(reader)
            i: int = 0
            for str in string_iterable_dataset:
                assert str == f"string_{i}"
                i = i + 1
        elif (reader.key.endswith('num-iterdataset-100.pkl')):
            assert reader.object_info.size == 62
        elif (reader.key.endswith('hello_world.txt')):
            assert reader.object_info.size == 14

if __name__ == '__main__':
    test_number_iterabledataset()
    test_string_iterabledataset()
    test_iterabledataset_reader()
    test_iterabledataset_from_bucket()

