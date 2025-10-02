#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import pytest
from unittest.mock import patch

import torch
import torch.distributed.checkpoint as dcp
import torchvision.models as models

from s3torchconnector import S3ReaderConstructor
from s3torchconnector.dcp import S3StorageWriter, S3StorageReader
from s3torchconnector.s3reader.sequential import SequentialS3Reader


@pytest.mark.parametrize(
    "model",
    [
        torch.nn.Sequential(
            torch.nn.Linear(5, 5),
            torch.nn.Linear(20, 20),
            torch.nn.Linear(10, 10),
        ),
        models.resnet18(pretrained=False),
    ],
)
def test_prepare_local_plan_sorts_by_storage_offset(checkpoint_directory, model):
    """
    Test that prepare_local_plan allows dcp.load() to read items in offset order.

    This does not prevent backwards seek, since torch.load() would still call
    backwards seek operations.

    pytorch/torch/serialization.py load() function will call _is_zipfile(), which
    includes this read() call: f.read(len(local_header_magic_number)). This is
    followed by readinto() calls on the actual tensor.

    Hence we can track read() call positions to determine if load ordering is
    being applied correctly.
    """
    region = checkpoint_directory.region
    s3_uri = checkpoint_directory.s3_uri

    state_dict = model.state_dict()
    storage_writer = S3StorageWriter(region=region, path=s3_uri, overwrite=True)
    dcp.save(state_dict, storage_writer=storage_writer)

    read_positions = []

    original_read = SequentialS3Reader.read

    def track_reads(self, size=None):
        if not self.key.endswith(".metadata"):
            read_positions.append(self._position)
        return original_read(self, size)

    # Load with position tracking on read() (called at the start of each torch.load())
    with patch.object(SequentialS3Reader, "read", track_reads):
        loaded_state_dict = {k: torch.empty_like(v) for k, v in state_dict.items()}
        storage_reader = S3StorageReader(
            region=region,
            path=s3_uri,
            reader_constructor=S3ReaderConstructor.sequential(),
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
