import logging

from s3dataset._s3dataset import MountpointS3Client

logging.basicConfig(format='%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s')
logging.getLogger().setLevel(1)

log = logging.getLogger(__name__)


def test_get_object():
    client = MountpointS3Client("us-east-1")
    stream = client.get_object("s3dataset-testing", "hello_world.txt")

    full_data = b''.join(stream)
    assert full_data == b"Hello, World!\n"


def test_list_objects():
    client = MountpointS3Client("us-east-1")
    stream = client.list_objects("s3dataset-testing")

    object_infos = [object_info for page in stream for object_info in page.object_info]
    keys = {object_info.key for object_info in object_infos}
    assert keys == {"hello_world.txt"}


if __name__ == "__main__":
    test_get_object()
    test_list_objects()
