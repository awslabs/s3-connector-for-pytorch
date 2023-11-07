from typing import Union

from s3dataset_s3_client._s3dataset import PutObjectStream


class PutObjectStreamWrapper:
    def __init__(self, stream: PutObjectStream):
        self.stream = stream

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def write(self, data: Union[bytes, memoryview]) -> int:
        if isinstance(data, memoryview):
            data = data.tobytes()
        self.stream.write(data)
        return len(data)

    def close(self):
        self.stream.close()

    def flush(self):
        pass
