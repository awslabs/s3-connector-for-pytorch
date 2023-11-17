#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import pickle

from hypothesis import given
from hypothesis.strategies import text

from s3torchconnectorclient import S3Exception


@given(text())
def test_pickles(message):
    exc = S3Exception(message)
    assert exc.args[0] == message
    unpickled = pickle.loads(pickle.dumps(exc))
    assert unpickled.args[0] == message


def test_multiple_arguments():
    args = ("foo", 1)
    exc = S3Exception(*args)
    assert exc.args == args
    unpickled = pickle.loads(pickle.dumps(exc))
    assert unpickled.args == args
