#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
import pytest
import torch
import torch.distributed.checkpoint as dcp

import torch.distributed as dist
from torch.distributed.checkpoint import CheckpointException
import torch.multiprocessing as mp

from s3torchconnector import S3ClientConfig
from s3torchconnector.dcp import (
    S3StorageWriter,
    S3StorageReader,
    S3FileSystem,
    BinaryPrefixStrategy,
)
from s3torchconnector.s3reader import (
    S3ReaderConstructor,
    S3ReaderConstructorProtocol,
)
from s3torchconnector._s3client import S3Client
from s3torchconnector._s3dataset_common import parse_s3_uri
from s3torchconnector._user_agent import UserAgent
from s3torchconnectorclient import __version__

import random
from typing import Optional

from s3torchconnector.dcp.s3_prefix_strategy import RoundRobinPrefixStrategy
from test_common import _list_folders_in_bucket

DEFAULT_USER_AGENT_PREFIX = UserAgent.get_default_prefix()


def generate_random_port():
    return random.randint(1, 500)


def setup(rank, world_size, port):
    dist.init_process_group(
        backend="gloo",
        world_size=world_size,
        rank=rank,
        init_method=f"tcp://127.0.0.1:{port}",
    )


def cleanup():
    # Synchronization point: Barrier ensures all process groups reach this point
    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


def run(
    rank,
    world_size,
    threads,
    region,
    s3_path_s3storagewriter,
    test_data,
    port,
    prefix_strategy,
):
    print(f"Running on rank {rank}.")

    setup(rank, world_size, port)
    # Save using S3StorageWriter
    dcp_save(
        test_data,
        S3StorageWriter(
            region=region,
            thread_count=threads,
            path=s3_path_s3storagewriter,
            overwrite=True,
            prefix_strategy=prefix_strategy,
        ),
    )

    cleanup()


def multi_process_dcp_save_load(
    world_size,
    thread_count,
    checkpoint_directory,
    tensor_dimensions,
    port_offset,
    prefix_strategy,
    reader_constructor=None,
) -> str:
    region = checkpoint_directory.region
    s3_path_s3storagewriter = f"{checkpoint_directory.s3_uri}checkpoint_s3storagewriter"
    s3_path_s3storagewriter = s3_path_s3storagewriter.replace("[", "_").replace(
        "]", "_"
    )

    test_data = {
        "tensor1": torch.randn(tensor_dimensions),
        "tensor2": torch.randn(5, 5),
        "scalar": torch.tensor(3.14),
    }

    port = str(generate_random_port() + port_offset)
    mp.spawn(
        run,
        args=(
            world_size,
            thread_count,
            region,
            s3_path_s3storagewriter,
            test_data,
            port,
            prefix_strategy,
        ),
        nprocs=world_size,
        join=True,
    )

    load_data(
        region,
        s3_path_s3storagewriter,
        test_data,
        world_size,
        thread_count,
        reader_constructor,
    )

    return s3_path_s3storagewriter


def _verify_user_agent(s3fs: S3FileSystem):
    reader_type_string = S3ReaderConstructor.get_reader_type_string(
        s3fs._reader_constructor
    )
    expected_user_agent = f"{DEFAULT_USER_AGENT_PREFIX} (dcp; {torch.__version__}; md/reader_type#{reader_type_string})"
    print(expected_user_agent)
    assert s3fs._client.user_agent_prefix == expected_user_agent


def dcp_save(data, writer):
    _verify_user_agent(writer.fs)
    dcp.save(
        data,
        storage_writer=writer,
    )


def dcp_load(loaded_data, reader):
    _verify_user_agent(reader.fs)
    dcp.load(
        loaded_data,
        storage_reader=reader,
    )


def load_data(
    region,
    s3_path_s3storagewriter,
    test_data,
    world_size,
    thread_count,
    reader_constructor: Optional[S3ReaderConstructorProtocol],
):
    s3_client = S3Client(region=region)
    bucket, key = parse_s3_uri(s3_path_s3storagewriter)
    list_result_s3storagewriter = list(s3_client.list_objects(bucket, f"{key}/"))

    # Compare length
    assert list_result_s3storagewriter is not None
    assert (
        len(list_result_s3storagewriter[0].object_info) == world_size * thread_count + 1
    )

    # Load using S3StorageReader
    loaded_data_s3storagereader = {}
    dcp_load(
        loaded_data_s3storagereader,
        S3StorageReader(
            region,
            s3_path_s3storagewriter,
            reader_constructor=reader_constructor,
        ),
    )

    for key in loaded_data_s3storagereader.keys():
        assert torch.allclose(
            loaded_data_s3storagereader[key], test_data[key]
        ), f"S3StorageReader: Loaded tensor for key '{key}' does not match original"

    print("Test passed: Saved and loaded data correctly.")


@pytest.mark.parametrize(
    "tensor_dimensions, thread_count, port_offset",
    [
        ([3, 2], 1, 10000),
        ([10, 1024, 1024], 1, 15000),
        ([3, 2], 4, 20000),
        ([10, 1024, 1024], 4, 25000),
    ],
    ids=[
        "small_tensor_single_thread",
        "large_tensor_single_thread",
        "small_tensor_multi_thread",
        "large_tensor_multi_thread",
    ],
)
def test_dcp_when_multi_process(
    checkpoint_directory,
    tensor_dimensions,
    thread_count,
    port_offset,
    reader_constructor,
):
    multi_process_dcp_save_load(
        world_size=3,
        thread_count=thread_count,
        checkpoint_directory=checkpoint_directory,
        tensor_dimensions=tensor_dimensions,
        port_offset=port_offset,
        prefix_strategy=None,
        reader_constructor=reader_constructor,
    )


@pytest.mark.parametrize("path", ["test_rename_src", "test_[+re.name]!@£$%^&*_(src)"])
def test_successful_rename(checkpoint_directory, path):
    src_path = f"{checkpoint_directory.s3_uri}{path}"
    _test_rename_internal(checkpoint_directory, src_path)
    if not checkpoint_directory.is_express_storage():
        # special case to test against buckets with dot in the name
        # S3 Express doesn't support such buckets names
        src_path.replace("cibucket", "cibucket.test")
        _test_rename_internal(checkpoint_directory, src_path)


def _test_rename_internal(checkpoint_directory, src_path):
    test_data = {
        "tensor1": torch.randn(10, 10),
        "tensor2": torch.randn(5, 5),
        "scalar": torch.tensor(3.14),
    }
    region = checkpoint_directory.region
    # Test S3StorageWriter
    s3_writer = S3StorageWriter(region, src_path, overwrite=False)
    dcp_save(test_data, s3_writer)
    s3_writer.fs.rename(f"{src_path}/.metadata", f"{src_path}/.metadata2")
    assert not s3_writer.fs.exists(f"{src_path}/.metadata")
    assert s3_writer.fs.exists(f"{src_path}/.metadata2")
    print("Test passed: Rename was successful.")


def test_rename_non_existing_s3_uri(checkpoint_directory):
    region = checkpoint_directory.region
    non_existing_s3_uri = f"{checkpoint_directory.s3_uri}non-existing-object"
    storage_writer = S3StorageWriter(region, non_existing_s3_uri, overwrite=True)

    with pytest.raises(Exception, match="Service error: The object was not found"):
        storage_writer.fs.rename(
            f"{non_existing_s3_uri}/.metadata", f"{non_existing_s3_uri}/.metadata2"
        )

    print("Test passed: Raised object not found error.")


def test_rm_file_non_existing_s3_uri(checkpoint_directory):
    region = checkpoint_directory.region
    non_existing_s3_uri = f"{checkpoint_directory.s3_uri}non-existing-object-hooo"
    storage_writer = S3StorageWriter(region, non_existing_s3_uri, overwrite=True)
    storage_writer.fs.rm_file(non_existing_s3_uri)

    print(
        "Test passed: In case of delete did not throw error if the object was not found."
    )


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


def test_s3client_config_for_writer(checkpoint_directory):
    region = checkpoint_directory.region
    s3_uri = checkpoint_directory.s3_uri
    s3client_config = S3ClientConfig(
        throughput_target_gbps=77, part_size=756, max_attempts=3
    )

    writer = S3StorageWriter(
        region, s3_uri, overwrite=True, s3client_config=s3client_config
    )
    assert (
        writer.fs._client._s3client_config.throughput_target_gbps
        == s3client_config.throughput_target_gbps
    )
    assert writer.fs._client._s3client_config.part_size == s3client_config.part_size
    assert (
        writer.fs._client._s3client_config.max_attempts == s3client_config.max_attempts
    )

    reader = S3StorageReader(region, s3_uri, s3client_config=s3client_config)
    assert (
        reader.fs._client._s3client_config.throughput_target_gbps
        == s3client_config.throughput_target_gbps
    )
    assert reader.fs._client._s3client_config.part_size == s3client_config.part_size
    assert (
        reader.fs._client._s3client_config.max_attempts == s3client_config.max_attempts
    )


@pytest.mark.parametrize(
    "tensor_dimensions, thread_count, port_offset",
    [
        ([1024, 10, 10], 1, 30000),
        ([1024, 10, 10], 4, 35000),
    ],
    ids=[
        "prefix_single_thread",
        "prefix_multi_thread",
    ],
)
def test_round_robin_prefix_strategy(
    checkpoint_directory, tensor_dimensions, thread_count, port_offset
):
    prefixes = ["prefix1", "prefix2", "prefix3"]
    rr_strategy = RoundRobinPrefixStrategy(prefixes)
    base_folder = multi_process_dcp_save_load(
        3,
        thread_count,
        checkpoint_directory,
        tensor_dimensions,
        port_offset,
        rr_strategy,
    )
    bucket, key = parse_s3_uri(base_folder)
    # ensure that provided prefixes were used for distributing checkpoints
    list_result = _list_folders_in_bucket(bucket, key)
    assert set(list_result) == set(prefixes)


@pytest.mark.parametrize(
    "tensor_dimensions, thread_count, port_offset",
    [
        ([1024, 10, 10], 1, 40000),
        ([1024, 10, 10], 4, 45000),
    ],
    ids=[
        "prefix_single_thread",
        "prefix_multi_thread",
    ],
)
def test_round_binary_prefix_strategy(
    checkpoint_directory, tensor_dimensions, thread_count, port_offset
):
    rr_strategy = BinaryPrefixStrategy(epoch_num=4, min_prefix_length=4, prefix_count=4)
    base_folder = multi_process_dcp_save_load(
        4,
        thread_count,
        checkpoint_directory,
        tensor_dimensions,
        port_offset,
        rr_strategy,
    )
    bucket, key = parse_s3_uri(base_folder)
    # ensure that provided prefixes were used for distributing checkpoints
    list_result = _list_folders_in_bucket(bucket, key)
    expected_prefixes = [
        "0000",
        "1000",
        "0100",
        "1100",
    ]
    assert set(list_result) == set(expected_prefixes)

    for partition_prefix in expected_prefixes:
        # ensure that sub folders for epochs were created
        list_result = _list_folders_in_bucket(bucket, f"{key}/{partition_prefix}")
        assert set(list_result) == {"epoch_4"}
