#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from contextlib import contextmanager
from unittest.mock import patch
import torch
from hypothesis.strategies import one_of, just

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
        return torch.load(fobj, weights_only=True)
