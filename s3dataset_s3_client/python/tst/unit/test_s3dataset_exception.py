import pickle

from hypothesis import given
from hypothesis.strategies import text

from s3dataset_s3_client import S3DatasetException


@given(text())
def test_pickles(message):
    exc = S3DatasetException(message)
    assert exc.args[0] == message
    unpickled = pickle.loads(pickle.dumps(exc))
    assert unpickled.args[0] == message
