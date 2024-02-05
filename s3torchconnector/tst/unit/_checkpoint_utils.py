#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from contextlib import contextmanager
from unittest.mock import patch
import torch

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


@contextmanager
def _patch_byteorder(byteorder: str):
    with patch("torch.serialization.sys") as mock_sys:
        mock_sys.byteorder = byteorder
        yield


def save_with_byteorder(data, fobj, byteorder: str, use_modern_pytorch_format: bool):
    with _patch_byteorder(byteorder):
        torch.save(data, fobj, _use_new_zipfile_serialization=use_modern_pytorch_format)


def load_with_byteorder(fobj, byteorder):
    with _patch_byteorder(byteorder):
        return torch.load(fobj)
