from io import BytesIO
from typing import Callable, Any

from hypothesis import given
from hypothesis.strategies import (
    integers,
    binary,
    none,
    characters,
    complex_numbers,
    floats,
    booleans,
    decimals,
    fractions,
    deferred,
    frozensets,
    tuples,
    dictionaries,
    lists,
    uuids,
    sets,
    text,
    just,
    one_of,
)
from torch import eq

from s3torchconnector._s3client import MockS3Client
from s3torchconnector import S3Checkpoint, S3LightningCheckpoint
from s3torchconnector.tst.unit.test_checkpointing import _load_with_byteorder

TEST_BUCKET = "test-bucket"
TEST_KEY = "test-key"
TEST_REGION = "us-east-1"

scalars = (
        none()
        | booleans()
        | integers()
        # Disallow nan as it doesn't have self-equality
        | floats(allow_nan=False)
        | complex_numbers(allow_nan=False)
        | decimals(allow_nan=False)
        | fractions()
        | characters()
        | binary(max_size=10)
        | text(max_size=10)
        | uuids()
)

hashable = deferred(
    lambda: (scalars | frozensets(hashable, max_size=5) | tuples(hashable))
)

python_primitives = deferred(
    lambda: (
            hashable
            | sets(hashable, max_size=5)
            | lists(python_primitives, max_size=5)
            | dictionaries(keys=hashable, values=python_primitives, max_size=3)
    )
)

byteorders = one_of(just("little"), just("big"))
use_modern_pytorch_format = booleans()


@given(python_primitives, byteorders, use_modern_pytorch_format)
def test_general_checkpointing_saves_python_primitives(
        data, byteorder
):
    _test_save(data, byteorder)


def _test_save(
        data,
        byteorder: str,
        *,
        equal: Callable[[Any, Any], bool] = eq,
):
    s3_lightning_checkpoint = S3LightningCheckpoint(TEST_REGION)

    # Use MockClient instead of actual client.
    client = MockS3Client(TEST_REGION, TEST_BUCKET)
    s3_lightning_checkpoint._client = client

    s3_lightning_checkpoint.save_checkpoint(data, f"s3://{TEST_BUCKET}/{TEST_KEY}")

    serialised = BytesIO(b"".join(client.get_object(TEST_BUCKET, TEST_KEY)))
    assert equal(_load_with_byteorder(serialised, byteorder), data)
