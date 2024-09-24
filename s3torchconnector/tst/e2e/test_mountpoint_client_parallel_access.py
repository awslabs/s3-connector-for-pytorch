import os
import random
import time
import threading
import pytest
from s3torchconnector._s3client import S3Client
from s3torchconnectorclient._mountpoint_s3_client import MountpointS3Client

class S3ClientWithoutLock(S3Client):
    @property
    def _client(self) -> MountpointS3Client:
        if self._client_pid is None or self._client_pid != os.getpid():
            self._client_pid = os.getpid()
            # `MountpointS3Client` does not survive forking, so re-create it if the PID has changed.
            self._real_client = self._client_builder()
        time.sleep(10)
        assert self._real_client is not None
        return self._real_client

    def invalidate_client(self):
        self._real_client = None

def access_client(client, error_event):
    try:
        if not error_event.is_set():
            client._client
            print(f"Successfully accessed by thread {threading.current_thread().name}")
    except AssertionError as e:
        print(f"AssertionError in thread {threading.current_thread().name}: {e}")
        error_event.set()

def invalidate_client(client, error_event):
    if not error_event.is_set():
        client.invalidate_client()
        print(f"Client invalidated by thread {threading.current_thread().name}")

def test_multiple_thread_accessing_mountpoint_client_in_parallel():
    print("Running test without lock...")
    client = S3ClientWithoutLock("us-west-2")
    error_event = threading.Event()

    # Start one accessor thread
    accessor_thread = threading.Thread(target=access_client, args=(client, error_event,), name="Accessor")
    accessor_thread.start()

    # Create and start multiple invalidator threads
    invalidator_threads = []
    num_invalidators = 500  # Number of invalidator threads

    for i in range(num_invalidators):
        if error_event.is_set():
            break
        invalidator_thread = threading.Thread(target=invalidate_client, args=(client, error_event,),
                                              name=f"Invalidator-{i + 1}")

        invalidator_threads.append(invalidator_thread)
        time.sleep(random.uniform(0.1, 0.5))
        invalidator_thread.start()

    accessor_thread.join()

    for thread in invalidator_threads:
        thread.join(timeout=1)

    if error_event.is_set():
        pytest.fail("Test failed due to AssertionError in one of the threads.")