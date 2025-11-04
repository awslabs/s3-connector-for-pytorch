#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import pytest
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.distributed.checkpoint as dcp

from s3torchconnector import S3ReaderConstructor
from s3torchconnector.dcp import S3StorageWriter, S3StorageReader
from s3torchconnector._s3client import S3Client

from ...conftest import READER_TYPE_STRING_TO_CLASS

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
def test_dcp_load_reads_tensors_in_sequential_order(
    dcp_reader_constructor, checkpoint_directory, model
):
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

    reader_type_string = S3ReaderConstructor.get_reader_type_string(
        dcp_reader_constructor
    )
    reader_class = READER_TYPE_STRING_TO_CLASS[reader_type_string]

    read_positions = []
    original_read = reader_class.read

    def track_reads(self, size=None):
        if not self.key.endswith(".metadata"):
            read_positions.append(self._position)
        return original_read(self, size)

    with patch.object(reader_class, "read", track_reads):
        loaded_state_dict = {k: torch.empty_like(v) for k, v in state_dict.items()}
        storage_reader = S3StorageReader(
            region=region, path=s3_uri, reader_constructor=dcp_reader_constructor
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


@pytest.mark.parametrize("model", [SIMPLE_MODEL, LARGER_MODEL])
@pytest.mark.parametrize(
    "max_gap_size,load_filter,filter_name,expected_streams",
    [
        # Full load - all tensors are consecutive, so always 1 stream
        (0, lambda k: True, "Full", 1),
        (float("inf"), lambda k: True, "Full", 1),
        # Weights only - scattered by biases, so stream count depends on max_gap_size
        (0, lambda k: k.endswith(".weight"), "Weights", 3),
        (float("inf"), lambda k: k.endswith(".weight"), "Weights", 1),
        # Layer 2 only - their bias+weight tensors are consecutive, so always 1 stream
        (0, lambda k: "2." in k, "Layer 2", 1),
        (float("inf"), lambda k: "2." in k, "Layer 2", 1),
    ],
)
def test_dcp_optimized_loading_patterns(
    checkpoint_directory,
    model,
    max_gap_size,
    load_filter,
    filter_name,
    expected_streams,
):
    """Test DCPOptimized reader with full and partial loading patterns and different max_gap_size.

    Validates that full loads use 1 stream, and partial load stream usage depends
    on max_gap_size and whether tensors are consecutive / neighbours.

    SIMPLE_MODEL tensors: ['0.bias', '0.weight', '1.bias', '1.weight', '2.bias', '2.weight']
    LARGER_MODEL tensors: ['linear_relu_stack.0.bias', 'linear_relu_stack.0.weight', 'linear_relu_stack.2.bias',
                           'linear_relu_stack.2.weight', 'linear_relu_stack.4.bias', 'linear_relu_stack.4.weight']
    """
    region = checkpoint_directory.region
    s3_uri = checkpoint_directory.s3_uri

    state_dict = model.state_dict()
    dcp.save(state_dict, storage_writer=S3StorageWriter(region, s3_uri, overwrite=True))

    # Print model structure (once per model)
    all_keys = list(state_dict.keys())
    if max_gap_size == 0 and filter_name == "Full":
        print(f"\nTensors: {sorted(all_keys)}")

    # Apply filter for partial load
    filtered_keys = [k for k in all_keys if load_filter(k)]
    excluded_keys = [k for k in all_keys if not load_filter(k)]
    assert filtered_keys, f"No keys match {filter_name} filter for this model"
    filtered_dict = {k: torch.empty_like(state_dict[k]) for k in filtered_keys}

    # Load full / partial checkpoint with stream call tracker
    stream_calls = []
    original_get_object_stream = S3Client._get_object_stream

    def track_get_object_stream(self, bucket, key, start=None, end=None):
        if not key.endswith(".metadata"):
            stream_calls.append((start, end))
        return original_get_object_stream(self, bucket, key, start=start, end=end)

    with patch.object(S3Client, "_get_object_stream", track_get_object_stream):
        reader_constructor = S3ReaderConstructor.dcp_optimized(max_gap_size)
        reader = S3StorageReader(region, s3_uri, reader_constructor=reader_constructor)
        dcp.load(filtered_dict, storage_reader=reader)

    # Verify correctness
    assert len(filtered_dict) == len(filtered_keys)
    for k, v in filtered_dict.items():
        assert torch.equal(v, state_dict[k])
        assert load_filter(k)

    # Verify excluded keys are not loaded
    for k in excluded_keys:
        assert k not in filtered_dict, f"Key {k} should not be in {filter_name} load"

    # Verify expected stream count
    assert len(stream_calls) == expected_streams
    if len(stream_calls) > 1:
        for i in range(1, len(stream_calls)):
            assert stream_calls[i][0] >= stream_calls[i - 1][1]
            assert stream_calls[i][0] - stream_calls[i - 1][1] >= max_gap_size

    # Print number of stream calls
    coalesce = "no coalesce" if max_gap_size == 0 else "full coalesce"
    print(
        f"{filter_name} load, {coalesce}: {len(stream_calls)} streams, {len(filtered_keys)} tensors"
    )
