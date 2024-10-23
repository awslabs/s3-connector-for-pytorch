#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import CheckpointException

from s3torchconnector.dcp import S3StorageWriter, S3StorageReader


def test_fsdp_filesystem_when_single_thread(checkpoint_directory):
    # TODO: implement me
    pass


def test_fsdp_filesystem_when_multiple_threads(checkpoint_directory):
    # TODO: implement me
    pass


# Inspired from https://github.com/pytorch/pytorch/blob/main/test/distributed/checkpoint/test_fsspec.py.
def test_overwrite(checkpoint_directory):
    t1, t2 = torch.randn(10), torch.randn(10)
    region = checkpoint_directory.region
    s3_uri = checkpoint_directory.s3_uri

    dcp.save(
        {"random": t1},
        storage_writer=S3StorageWriter(region, s3_uri, overwrite=False),
    )
    dcp.save(
        {"random": t2},
        storage_writer=S3StorageWriter(region, s3_uri, overwrite=True),
    )

    sd = {"random": torch.zeros(10)}
    dcp.load(sd, checkpoint_id=s3_uri, storage_reader=S3StorageReader(region, s3_uri))
    assert torch.allclose(sd["random"], t2) is True

    with pytest.raises(CheckpointException) as excinfo:
        dcp.save(
            {"random": t2},
            storage_writer=S3StorageWriter(region, s3_uri, overwrite=False),
        )

    assert "Checkpoint already exists" in str(excinfo.value)
