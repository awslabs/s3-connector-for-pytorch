import os
import time
import threading
import pytest
from s3torchconnector._s3client import S3Client
from s3torchconnectorclient._mountpoint_s3_client import MountpointS3Client

from test_common import _get_fork_methods
from conftest import getenv

NATIVE_S3_CLIENT = None


class S3ClientWithoutLock(S3Client):
    @property
    def _client(self) -> MountpointS3Client:
        global NATIVE_S3_CLIENT
        if self._client_pid is None or self._client_pid != os.getpid():
            self._client_pid = os.getpid()
            # `MountpointS3Client` does not survive forking, so re-create it if the PID has changed.
            NATIVE_S3_CLIENT = self._client_builder()
        assert NATIVE_S3_CLIENT is not None
        return NATIVE_S3_CLIENT

    def _client_builder(self):
        time.sleep(1)
        return super()._client_builder()


class S3ClientWithLock(S3Client):
    def _client_builder(self):
        time.sleep(1)
        return super()._client_builder()


def test_s3_client_reset_after_fork():
    methods = _get_fork_methods()
    if "fork" not in methods:
        pytest.skip("fork is not supported")
    region = getenv("CI_REGION")
    s3_client1 = S3Client(region=region)
    s3_client2 = S3Client(region=region)

    if (
        s3_client1._client is None
        or s3_client2._client is None
        or s3_client1._native_client is None
        or s3_client2._native_client is None
    ):
        pytest.fail("Native client is not initialized")
    # fork process to clean-up clients
    # Fork process
    pid = os.fork()

    if pid == 0:
        # Child process
        try:
            if (
                s3_client1._native_client is not None
                or s3_client2._native_client is not None
            ):
                os._exit(1)  # Fail if clients are not reset
            os._exit(0)  # Success
        finally:
            os._exit(1)  # Ensure child exits in case of any other error
    else:
        # Parent process
        _, status = os.waitpid(pid, 0)
        exit_code = os.WEXITSTATUS(status)

        if (
            exit_code != 0
            or s3_client1._native_client is not None
            or s3_client2._native_client is not None
        ):
            pytest.fail("Native client is not reset after fork")


def access_client(client, error_event):
    try:
        if not error_event.is_set():
            client._client
            print(f"Successfully accessed by thread {threading.current_thread().name}")
    except AssertionError as e:
        print(f"AssertionError in thread {threading.current_thread().name}: {e}")
        error_event.set()


def test_multiple_thread_accessing_mountpoint_client_in_parallel_without_lock():
    print("Running test without lock...")
    client = S3ClientWithoutLock("us-west-2")
    if not access_mountpoint_client_in_parallel(client):
        pytest.fail(
            "Test failed as AssertionError did not happen in one of the threads."
        )


def test_multiple_thread_accessing_mountpoint_client_in_parallel_with_lock():
    print("Running test with lock...")
    client = S3ClientWithLock("us-west-2")
    if access_mountpoint_client_in_parallel(client):
        pytest.fail("Test failed as AssertionError happened in one of the threads.")


def access_mountpoint_client_in_parallel(client):

    error_event = threading.Event()
    # Create and start multiple threads
    accessor_threads = []
    num_accessor_threads = 10

    for i in range(num_accessor_threads):
        if error_event.is_set():
            break
        accessor_thread = threading.Thread(
            target=access_client,
            args=(
                client,
                error_event,
            ),
            name=f"Accessor-{i + 1}",
        )
        accessor_threads.append(accessor_thread)
        accessor_thread.start()

    for thread in accessor_threads:
        thread.join()

    return error_event.is_set()
