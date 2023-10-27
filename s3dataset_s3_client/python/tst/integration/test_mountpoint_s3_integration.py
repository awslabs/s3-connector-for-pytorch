import logging
import pickle

from s3dataset_s3_client._s3dataset import MountpointS3Client

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


def test_get_object_with_unpickled_client():
    original_client = MountpointS3Client("eu-west-2")
    pickled_client = pickle.dumps(original_client)
    assert isinstance(pickled_client, bytes)
    unpickled_client = pickle.loads(pickled_client)

    stream = unpickled_client.get_object(
        "dataset-it-bucket",
        "sample-files/hello_world.txt",
    )
    full_data = b"".join(stream)
    assert full_data == b"Hello, World!\n"


def test_list_objects():
    client = MountpointS3Client("eu-west-2")
    stream = client.list_objects("dataset-it-bucket")

    object_infos = [object_info for page in stream for object_info in page.object_info]
    keys = [object_info.key for object_info in object_infos]
    assert len(keys) > 1

    e2e_img_10_keys = [key for key in keys if key.startswith("e2e-tests/images-10/img")]
    expected_img_10_keys = [f"e2e-tests/images-10/img{i}.jpg" for i in range(10)]
    assert e2e_img_10_keys == expected_img_10_keys


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
