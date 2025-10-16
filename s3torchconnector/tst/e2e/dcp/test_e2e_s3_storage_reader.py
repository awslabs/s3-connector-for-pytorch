#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import pytest
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.distributed.checkpoint as dcp

from s3torchconnector import S3ReaderConstructor
from s3torchconnector.dcp import S3StorageWriter, S3StorageReader
from s3torchconnector.s3reader import SequentialS3Reader, DCPOptimizedS3Reader

SIMPLE_MODEL = torch.nn.Sequential(
    nn.Linear(5, 5),
    nn.Linear(20, 20),
    nn.Linear(10, 10),
)


class NeuralNetwork(nn.Module):
    """NeuralNetwork from PyTorch quickstart tutorial."""

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )


LARGER_MODEL = NeuralNetwork()


@pytest.mark.parametrize("model", [SIMPLE_MODEL, LARGER_MODEL])
@pytest.mark.parametrize(
    "reader_class,reader_constructor",
    [
        (SequentialS3Reader, S3ReaderConstructor.sequential()),
        (DCPOptimizedS3Reader, S3ReaderConstructor.dcp_optimized()),
    ],
)
def test_dcp_load_reads_tensors_in_sequential_order(
    checkpoint_directory, model, reader_class, reader_constructor
):
    """
    Test that prepare_local_plan allows dcp.load() to read items in offset order.

    This does not prevent backwards seek, since torch.load() would still call
    backwards seek operations.

    SequentialS3Reader:
    pytorch/torch/serialization.py load() function will call _is_zipfile(), which
    includes this read() call: f.read(len(local_header_magic_number)). This is
    followed by readinto() calls on the actual tensor.

    DCPOptimizedS3Reader:
    DCPOptimizedS3Reader.seekable() returns false, hence PyTorch would use read()
    calls and make it seekable with `seekable = io.BytesIO(transform_from.read(-1))` in
    pytorch/torch/distributed/checkpoint/filesystem.py read_data() method.

    Hence we can track read() call positions to determine if load ordering is
    being applied correctly for both cases.
    """
    region = checkpoint_directory.region
    s3_uri = checkpoint_directory.s3_uri

    state_dict = model.state_dict()
    storage_writer = S3StorageWriter(region=region, path=s3_uri, overwrite=True)
    dcp.save(state_dict, storage_writer=storage_writer)

    read_positions = []
    original_read = reader_class.read

    def track_reads(self, size=None):
        if not self.key.endswith(".metadata"):
            read_positions.append(self._position)
        return original_read(self, size)

    with patch.object(reader_class, "read", track_reads):
        loaded_state_dict = {k: torch.empty_like(v) for k, v in state_dict.items()}
        storage_reader = S3StorageReader(
            region=region, path=s3_uri, reader_constructor=reader_constructor
        )
        dcp.load(loaded_state_dict, storage_reader=storage_reader)

    print(f"Read positions: {read_positions}")

    # Assert load ordering works (read() calls should be in sorted order)
    assert read_positions == sorted(read_positions)

    # Assert all tensors are correctly loaded
    assert len(loaded_state_dict) == len(state_dict)
    assert loaded_state_dict.keys() == state_dict.keys()
    for key in state_dict:
        assert torch.equal(loaded_state_dict[key], state_dict[key])
