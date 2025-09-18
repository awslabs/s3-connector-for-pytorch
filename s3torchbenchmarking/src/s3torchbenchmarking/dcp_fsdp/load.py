#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import logging
import functools
from time import perf_counter
import os
import argparse

from s3torchconnector._s3client.s3client_config import S3ClientConfig
import torch.distributed.checkpoint as dcp
import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from s3torchconnector.dcp import S3StorageReader
from s3torchbenchmarking.models import get_benchmark_model
from s3torchbenchmarking.benchmark_utils import build_checkpoint_uri

logger = logging.getLogger(__name__)


def get_reader(region: str, uri: str, suffix: str) -> S3StorageReader:
    uri = build_checkpoint_uri(uri, suffix)
    logger.info("Loading checkpoint from %s (S3)...", uri)
    return S3StorageReader(
        region,
        uri,
        s3client_config=S3ClientConfig(
            part_size=5 * 1024 * 1024, throughput_target_gbps=300
        ),
    )


@record
def run_fsdp_load(
    rank,
    world_size,
    backend,
    region,
    uri,
    suffix,
    model_name="L7b",
    checkpoint_sharding_strategy="hybrid",
):
    if rank == 0:
        logger.info("Creating model")
        model_proxy = get_benchmark_model(model_name)
        model = model_proxy.model
    else:
        with torch.device("meta"):
            model_proxy = get_benchmark_model(model_name)
            model = model_proxy.model

    transformer_layer = LlamaDecoderLayer
    gpt_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={transformer_layer},
    )

    if backend == "nccl":
        device_id = rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        param_init_fn = lambda module: module.to_empty(
            device=torch.device("cuda"), recurse=False
        )
    else:
        device_id = rank % torch.cpu.device_count()
        torch.cpu.set_device(device_id)
        param_init_fn = lambda module: module.to_empty(
            device=torch.device("cpu"), recurse=False
        )

    sharding_strategy = (
        ShardingStrategy.HYBRID_SHARD
        if checkpoint_sharding_strategy == "hybrid"
        else ShardingStrategy.FULL_SHARD
    )

    model = FSDP(
        model,
        auto_wrap_policy=gpt_auto_wrap_policy,
        device_id=(
            torch.cuda.current_device()
            if backend == "nccl"
            else torch.cpu.current_device()
        ),
        use_orig_params=False,
        sharding_strategy=sharding_strategy,
        sync_module_states=True if backend == "nccl" else False,
        param_init_fn=param_init_fn if rank != 0 else None,
    )

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        state_dict = {"model": model.state_dict()}

    if rank == 0:
        print(f"Loading checkpoint with suffix: {suffix}")

    start_load = perf_counter()
    storage_reader = get_reader(region, uri, suffix)
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        dcp.load(state_dict, storage_reader=storage_reader)
    end_load = perf_counter()

    if rank == 0:
        print(f"Load time: {end_load - start_load:.3f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"])
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--uri", type=str, required=True)
    parser.add_argument("--suffix", type=str, required=True)
    parser.add_argument("--model", type=str, default="L7b")
    args = parser.parse_args()

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    if not dist.is_initialized():
        dist.init_process_group(args.backend, rank=rank, world_size=world_size)

    run_fsdp_load(
        rank, world_size, args.backend, args.region, args.uri, args.suffix, args.model
    )

    if dist.is_initialized():
        dist.destroy_process_group()
