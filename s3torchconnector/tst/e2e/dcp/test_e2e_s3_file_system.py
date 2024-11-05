#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import pytest
import torch
import torch.distributed.checkpoint as dcp

import torch.distributed as dist
from torch.distributed.checkpoint import CheckpointException
from torch.distributed.checkpoint._fsspec_filesystem import (
    FsspecWriter,
    FsspecReader,
)
import torch.multiprocessing as mp

from s3torchconnector.dcp import S3StorageWriter, S3StorageReader
from s3torchconnector._s3client import S3Client
from s3torchconnector._s3dataset_common import parse_s3_uri
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run(rank, world_size, threads, region, s3_path_s3storagewriter, s3_path_fsspec, test_data):
    print(f"Running on rank {rank}.")

    setup(rank, world_size)
    # Save using S3StorageWriter
    dcp_save(test_data, S3StorageWriter(region=region, thread_count=threads, path=s3_path_s3storagewriter, overwrite=True, ),)
    # Save using FsspecWriter
    dcp_save(test_data, FsspecWriter(path=s3_path_fsspec, overwrite=True, sync_files=False, ),)
    cleanup()

def multi_process_dcp_save_load(world_size, thread_count, checkpoint_directory):
    region = checkpoint_directory.region
    s3_path_s3storagewriter = f"{checkpoint_directory.s3_uri}checkpoint_s3storagewriter"
    s3_path_fsspec = f"{checkpoint_directory.s3_uri}checkpoint_fsspec"
    test_data = {
        "tensor1": torch.randn(10, 10),
        "tensor2": torch.randn(5, 5),
        "scalar": torch.tensor(3.14)
    }
    mp.spawn(run,
             args=(world_size, thread_count, region, s3_path_s3storagewriter, s3_path_fsspec, test_data),
             nprocs=world_size,
             join=True)

    load_data(region, s3_path_s3storagewriter, s3_path_fsspec, test_data, world_size, thread_count)

def dcp_save(data, writer):
    dcp.save(
        data,
        storage_writer=writer,
    )

def dcp_load(loaded_data, reader):
    dcp.load(
        loaded_data,
        storage_reader=reader,
    )

def load_data(region, s3_path_s3storagewriter, s3_path_fsspec, test_data, world_size, thread_count):
    s3_client = S3Client(region=region)
    bucket, key = parse_s3_uri(s3_path_s3storagewriter)
    list_result_s3storagewriter = list(s3_client.list_objects(bucket, key))

    # Compare length
    assert list_result_s3storagewriter is not None
    assert len(list_result_s3storagewriter[0].object_info) == world_size * thread_count + 1

    # Load using S3StorageReader
    loaded_data_s3storagereader = {}
    dcp_load(loaded_data_s3storagereader, S3StorageReader(region, s3_path_s3storagewriter,),)

    # Load using FsspecReader
    loaded_data_fsspec = {}
    dcp_load(loaded_data_fsspec, FsspecReader(s3_path_fsspec,),)

    # Compare loaded data
    assert set(loaded_data_s3storagereader.keys()) == set(loaded_data_fsspec.keys()), "Loaded data keys do not match"

    for key in loaded_data_s3storagereader.keys():
        assert torch.allclose(loaded_data_s3storagereader[key],
                              loaded_data_fsspec[key]), f"Loaded tensors for key '{key}' do not match"
        assert torch.allclose(loaded_data_s3storagereader[key],
                              test_data[key]), f"S3StorageReader: Loaded tensor for key '{key}' does not match original"
        assert torch.allclose(loaded_data_fsspec[key],
                              test_data[key]), f"FsspecReader: Loaded tensor for key '{key}' does not match original"

    print("Test passed: Both implementations saved and loaded data correctly, and the contents match.")


def test_dcp_when_multi_process_single_thread(checkpoint_directory):
    multi_process_dcp_save_load(6, 1, checkpoint_directory)

def test_dcp_when_multi_process_multiple_threads(checkpoint_directory):
    multi_process_dcp_save_load(6, 4, checkpoint_directory)

def test_dcp_save_non_existing_s3_uri(checkpoint_directory):
    t1 = torch.randn(10)
    region = checkpoint_directory.region
    non_existing_s3_uri = "s3://non-existing-bucket/checkpoint"

    with pytest.raises(CheckpointException) as s3_excinfo:
        dcp_save({"random": t1}, S3StorageWriter(region, non_existing_s3_uri, overwrite=True,),)

    with pytest.raises(CheckpointException) as fsspec_excinfo:
        dcp_save({"random": t1}, FsspecWriter(non_existing_s3_uri, overwrite=True, sync_files=False,),)

    # Assert that both exceptions are instances of CheckpointException
    assert isinstance(s3_excinfo.value, CheckpointException), "Using S3StorageWriter DCP should raise a CheckpointException"
    assert isinstance(fsspec_excinfo.value, CheckpointException), "Using FsspecWriter DCP should raise a CheckpointException"

    print("Test passed: Both S3StorageWriter and FsspecWriter implementations raised the same CheckpointException.")

def test_dcp_load_non_existing_s3_uri(checkpoint_directory):
    region = checkpoint_directory.region
    non_existing_s3_uri= "s3://non-existing-bucket/checkpoint"

    with pytest.raises(CheckpointException) as s3_excinfo:
        dcp_load({}, S3StorageReader(region, non_existing_s3_uri,),)

    with pytest.raises(CheckpointException) as fsspec_excinfo:
        dcp_load({}, FsspecReader(non_existing_s3_uri,),)

    # Assert that both exceptions are instances of CheckpointException
    assert isinstance(s3_excinfo.value, CheckpointException), "Using S3StorageReader DCP should raise a CheckpointException"
    assert isinstance(fsspec_excinfo.value, CheckpointException), "Using FsspecReader DCP should raise a CheckpointException"

    print("Test passed: Both S3StorageReader and FsspecReader implementations raised the same CheckpointException.")

def test_successful_rename(checkpoint_directory):
    src_path = f"{checkpoint_directory.s3_uri}test_rename_src"
    test_data = {
        "tensor1": torch.randn(10, 10),
        "tensor2": torch.randn(5, 5),
        "scalar": torch.tensor(3.14)
    }
    region = checkpoint_directory.region

    # Test S3StorageWriter
    s3_writer = S3StorageWriter(region, src_path, overwrite=False)
    dcp_save(test_data, s3_writer)
    s3_writer.fs.rename(f"{src_path}/.metadata", f"{src_path}/.metadata2")

    assert not s3_writer.fs.exists(f"{src_path}/.metadata")
    assert s3_writer.fs.exists(f"{src_path}/.metadata2")

    s3_writer.fs.rm_file(f"{src_path}/.metadata2")

    # Test FsspecWriter
    fsspec_writer = FsspecWriter(src_path, overwrite=True, sync_files=False,)
    dcp_save(test_data, fsspec_writer)
    fsspec_writer.fs.rename(f"{src_path}/.metadata", f"{src_path}/.metadata2")

    assert not fsspec_writer.fs.exists(f"{src_path}/.metadata")
    assert fsspec_writer.fs.exists(f"{src_path}/.metadata2")

    print("Test passed: Rename behavior is same for both S3StorageWriter and FsspecWriter implementations.")

def test_rename_non_existing_s3_uri(checkpoint_directory):
    region = checkpoint_directory.region
    non_existing_s3_uri = f"{checkpoint_directory.s3_uri}non-existing-object"
    storage_writer = S3StorageWriter(region, non_existing_s3_uri, overwrite=True)
    storage_writer_fsspec = FsspecWriter(non_existing_s3_uri, overwrite=True, sync_files=False)
    with pytest.raises(Exception, match="Service error: The object was not found"):
        storage_writer.fs.rename(f"{non_existing_s3_uri}/.metadata", f"{non_existing_s3_uri}/.metadata2")
    with pytest.raises(FileNotFoundError):
        storage_writer_fsspec.fs.rename(f"{non_existing_s3_uri}/.metadata", f"{non_existing_s3_uri}/.metadata2")

    print("Test passed: Both S3StorageWriter and FsspecWriter implementations raised similar kind of exception.")

def test_rm_file_non_existing_s3_uri(checkpoint_directory):
    region = checkpoint_directory.region
    non_existing_s3_uri = f"{checkpoint_directory.s3_uri}non-existing-object-hooo"
    storage_writer = S3StorageWriter(region, non_existing_s3_uri, overwrite=True)
    storage_writer_fsspec = FsspecWriter(non_existing_s3_uri, overwrite=True, sync_files=False)
    storage_writer.fs.rm_file(non_existing_s3_uri)
    storage_writer_fsspec.fs.rm_file(non_existing_s3_uri)

    print("Test passed: Both S3StorageWriter and FsspecWriter implementations in case of delete do not throw error if the object was not found.")

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
