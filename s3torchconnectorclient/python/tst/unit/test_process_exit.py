#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import subprocess
import sys
import textwrap

import pytest

# A process whose client outlives the interpreter takes about a second to exit, so this leaves
# headroom for slow CI hosts.
EXIT_TIMEOUT_SECONDS = 10

DAEMON_THREAD_HOLDS_CLIENT = """
    import threading, time
    from s3torchconnectorclient._mountpoint_s3_client import MountpointS3Client

    def keep_alive(client):
        while True:
            time.sleep(0.05)

    client = MountpointS3Client(region="us-east-1", unsigned=True)
    threading.Thread(target=keep_alive, args=(client,), daemon=True).start()
    del client
"""

DAEMON_THREAD_KEEPS_GLOBALS_ALIVE = """
    import threading, time
    from s3torchconnectorclient._mountpoint_s3_client import MountpointS3Client

    client = MountpointS3Client(region="us-east-1", unsigned=True)

    def heartbeat():
        # Never touches the client, but its frame keeps this module's globals alive.
        while True:
            time.sleep(0.05)

    threading.Thread(target=heartbeat, daemon=True).start()
"""


@pytest.mark.parametrize(
    "script",
    [DAEMON_THREAD_HOLDS_CLIENT, DAEMON_THREAD_KEEPS_GLOBALS_ALIVE],
    ids=["daemon_thread_holds_client", "daemon_thread_keeps_globals_alive"],
)
def test_process_exits_while_daemon_thread_keeps_client_alive(script: str):
    # CPython never unwinds daemon threads at shutdown, so the client is never freed and its
    # CRT threads keep running. The CRT cleanup at process exit must not wait for them forever.
    try:
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            capture_output=True,
            text=True,
            timeout=EXIT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"process did not exit within {EXIT_TIMEOUT_SECONDS}s")
    assert result.returncode == 0, result.stderr
