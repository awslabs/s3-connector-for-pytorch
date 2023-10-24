import logging

from s3dataset._s3dataset import MountpointS3Client

logging.basicConfig(
    format="%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s"
)
logging.getLogger().setLevel(1)

log = logging.getLogger(__name__)


def test_get_object():
    client = MountpointS3Client("eu-west-2")
    stream = client.get_object("dataset-it-bucket", "sample-files/hello_world.txt")

    full_data = b"".join(stream)
    assert full_data == b"Hello, World!\n"


def test_list_objects():
    client = MountpointS3Client("eu-west-2")
    stream = client.list_objects("dataset-it-bucket")

    object_infos = [object_info for page in stream for object_info in page.object_info]
    keys = {object_info.key for object_info in object_infos}
    assert keys == {
        "iterable-datasets/",
        "iterable-datasets/num-iterdataset-100.pkl",
        "iterable-datasets/pickle-num-iterdataset-100.pkl",
        "iterable-datasets/pickle-str-iterdataset-10.pkl",
        "iterable-datasets/str-iterdataset-10.pkl",
        "iterable-datasets/torch-num-iterdataset-100.pkl",
        "iterable-datasets/torch-str-iterdataset-10.pkl",
        "sample-files/",
        "sample-files/catalog.csv",
        "sample-files/hello_world.txt",
    }


def test_list_objects_with_prefix():
    client = MountpointS3Client("eu-west-2")
    stream = client.list_objects("dataset-it-bucket", "sample-files/")

    object_infos = [object_info for page in stream for object_info in page.object_info]
    keys = {object_info.key for object_info in object_infos}
    assert keys == {
        "sample-files/",
        "sample-files/catalog.csv",
        "sample-files/hello_world.txt",
    }
