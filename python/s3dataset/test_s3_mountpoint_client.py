import logging

from s3dataset._s3dataset import MountpointS3Client

logging.basicConfig(format='%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s')
logging.getLogger().setLevel(1)

log = logging.getLogger(__name__)


def main():
    client = MountpointS3Client("us-east-1")
    stream = client.get_object("s3dataset-testing", "hello_world.txt")

    full_data = b''.join(stream)
    assert full_data == b"Hello, World!\n"


if __name__ == "__main__":
    main()
