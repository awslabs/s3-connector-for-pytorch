#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
import atexit
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple
import os
import hydra
import torchdata  # type: ignore
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, default_collate
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torchdata.datapipes.utils import StreamWrapper  # type: ignore
from s3torchbenchmarking.benchmark_utils import ExperimentResult
from s3torchbenchmarking.models import (
    Entitlement,
    ViT,
    ModelInterface,
)

from s3torchconnector import S3MapDataset, S3Reader, S3IterableDataset
from s3torchconnector.s3reader import S3ReaderConstructor, S3ReaderConstructorProtocol
from s3torchconnector._s3dataset_common import parse_s3_uri  # type: ignore
import torch
import logging
import torch.multiprocessing as mp
import json
import tempfile

logger = logging.getLogger(__name__)


def init_distributed(rank=0, world_size=1):
    """Initialize DDP Process group"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


# TODO: add Structured Config (https://hydra.cc/docs/tutorials/structured_config/intro/)
@hydra.main(version_base=None)
def run_experiment(config: DictConfig) -> dict:

    num_gpus = (
        config.num_gpus if hasattr(config, "num_gpus") else torch.cuda.device_count()
    )
    # Cap number of GPUs to max number of GPUs
    num_gpus = min(num_gpus, torch.cuda.device_count())
    # Set visible devices to limit GPU usage, this allows calls to torch.cuda.device_count
    # to use the inputted GPU count
    if num_gpus < torch.cuda.device_count():
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(num_gpus)))
        torch.cuda.empty_cache()

    # In the case of multiple GPU training we run run_ddp_process using mp.spawn for each of the ranks, which calls run_benchmark_experiment separately
    # If single-rank training then it defaults to run_benchmark_experiment
    if num_gpus > 1 and not dist.is_initialized():
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            results_file = f.name
        try:
            mp.spawn(
                run_ddp_process,
                args=(num_gpus, config, results_file),
                nprocs=num_gpus,
                join=True,
            )
            # Read results from file
            if os.path.exists(results_file):
                with open(results_file, "r") as f:
                    return json.load(f)
            else:
                return {"metrics": "DDP training completed - no results file"}

        finally:
            # Cleanup
            if os.path.exists(results_file):
                os.unlink(results_file)
    else:
        return run_benchmark_experiment(config)


# DDP Process function for running only in multi-GPU cases
def run_ddp_process(rank, world_size, config, results_file):

    init_distributed(rank, world_size)
    try:
        result = run_benchmark_experiment(config)

        # Gather metrics from all ranks
        all_metrics = [None] * world_size
        dist.all_gather_object(all_metrics, result["metrics"])

        # Only rank 0 aggregates and writes results
        if rank == 0 and result:
            # Aggregate metrics
            total_volume = sum(m.get("volume_mibs", 0) for m in all_metrics)
            avg_training_duration = (
                sum(m["training_duration_s"] for m in all_metrics) / world_size
            )

            aggregated_metrics = {
                "throughput_mibs": (
                    total_volume / avg_training_duration
                    if avg_training_duration > 0
                    else 0
                ),
                "training_duration_s": avg_training_duration,
                "volume_mibs": total_volume,
                "per_rank_metrics": all_metrics,
            }

            with open(results_file, "w") as f:
                json.dump({"metrics": aggregated_metrics}, f)
    finally:
        dist.destroy_process_group()


def run_benchmark_experiment(config: DictConfig):

    rank = dist.get_rank() if dist.is_initialized() else 0
    model = make_model(config)

    fully_qualified_uri = (
        "s3://" + config.s3.bucket.strip("/") + "/" + config.dataset.strip("/") + "/"
    )
    # We always return sample from make_dataset which could be None
    dataset, sampler = make_dataset(
        dataloader_config=config.dataloader,
        sharding=config.sharding,
        prefix_uri=fully_qualified_uri,
        region=config.s3.region,
        load_sample=model.load_sample,
    )
    dataloader = make_dataloader(
        dataset=dataset,
        sampler=sampler,
        num_workers=config.dataloader.num_workers,
        batch_size=config.dataloader.batch_size,
    )
    if dist.is_available() and dist.is_initialized():
        torch.cuda.set_device(rank)
        device_id = torch.cuda.current_device()

        if model.model != None:
            model.model = model.model.to(device_id)
            model.device = torch.device(f"cuda:{device_id}")
            model.model = torch.nn.parallel.DistributedDataParallel(
                model.model, device_ids=[rank], output_device=rank
            )
            # Recreate optimizer AFTER DDP wrapping
            model._optimizer = None  # Clear cached optimizer
            model.optimizer = torch.optim.Adam(model.model.parameters(), lr=0.001)

    result: ExperimentResult = model.train(dataloader, config.epochs)

    metrics = {
        "throughput_mibs": result["volume"] / result["training_duration_s"],
        "training_duration_s": result["training_duration_s"],
        "volume_mibs": result[
            "volume"
        ],  # Includes number of samples for context on how many images were processed in total
        "epoch_durations_s": result["epoch_durations_s"],
        "utilization": {k: v.summarize() for k, v in result["utilization"].items()},
    }
    return {"metrics": metrics}


def make_model(config: DictConfig) -> ModelInterface:
    if config.model == "entitlement":
        return Entitlement()
    elif config.model == "vit":
        num_labels = int(config.get("num_labels", 1000))
        return ViT(num_labels, config.checkpoint)
    else:
        raise Exception(f"Unknown model {config.model}")


def make_mountpoint(
    prefix_uri: str,
    mountpoint_path: Optional[str] = None,
    additional_args: Optional[List[str]] = None,
) -> str:
    def teardown(path: str):
        subprocess.run(["sudo", "umount", path])
        shutil.rmtree(path)

    bucket, prefix = parse_s3_uri(prefix_uri)
    # Run Mountpoint in background mode, and arrange for it to unmount when this script exits
    tempdir = tempfile.mkdtemp(prefix="s3dataset_")
    binary = mountpoint_path or "mount-s3"
    args = additional_args or []
    subprocess.run([binary, bucket, tempdir] + args, check=True)
    atexit.register(teardown, tempdir)

    # Now we can just read our dataset as if it were a local directory
    return str(Path(tempdir) / prefix)


def make_dataset(
    dataloader_config: DictConfig,
    sharding: bool,
    prefix_uri: str,
    region: Optional[str],
    load_sample,
) -> Tuple[Dataset, Optional[DistributedSampler]]:

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    kind = dataloader_config.kind
    num_workers = dataloader_config.num_workers
    if kind == "s3iterabledataset":
        if not region:
            raise ValueError("Must provide region for s3iterabledataset")
        if not dataloader_config.get("s3reader"):
            raise ValueError(f"Must provide s3reader config for {kind}")
        s3reader_config = dataloader_config.s3reader
        return (
            create_s3_iterable_dataset(
                sharding,
                prefix_uri,
                region,
                load_sample,
                num_workers,
                s3reader_config,
                world_size,
            ),
            None,
        )
    if kind == "s3mapdataset":
        if not region:
            raise ValueError("Must provide region for s3mapdataset")
        if not dataloader_config.get("s3reader"):
            raise ValueError(f"Must provide s3reader config for {kind}")
        s3reader_config = dataloader_config.s3reader
        return create_s3_map_dataset(
            sharding, prefix_uri, region, load_sample, s3reader_config, world_size, rank
        )
    if kind == "fsspec":
        return (
            create_fsspec_dataset(sharding, prefix_uri, load_sample, num_workers),
            None,
        )
    if kind == "mountpoint":
        return (
            create_mountpoint_dataset(
                sharding, prefix_uri, load_sample, num_workers, False
            ),
            None,
        )
    if kind == "mountpointcache":
        return (
            create_mountpoint_dataset(
                sharding, prefix_uri, load_sample, num_workers, True
            ),
            None,
        )
    raise Exception(f"Unknown dataset kind {kind}")


def make_s3_reader_constructor(
    s3reader_config: DictConfig,
) -> S3ReaderConstructorProtocol:
    s3reader_type = s3reader_config.type
    if s3reader_type == "sequential":
        reader_constructor = S3ReaderConstructor.sequential()
    elif s3reader_type == "range_based":
        buffer_size_value = s3reader_config.buffer_size
        if isinstance(buffer_size_value, str):
            # Safely evaluate simple math expressions (remove access to dangerous functions)
            buffer_size = int(eval(buffer_size_value, {"__builtins__": {}}, {}))
        else:
            buffer_size = int(buffer_size_value)
        reader_constructor = S3ReaderConstructor.range_based(buffer_size=buffer_size)
    else:
        raise ValueError(f"Unknown s3reader type {s3reader_type}")

    return reader_constructor


def create_s3_iterable_dataset(
    sharding: bool,
    prefix_uri: str,
    region: str,
    load_sample,
    num_workers: int,
    s3reader_config: DictConfig,
    world_size: int = 1,
):
    reader_constructor = make_s3_reader_constructor(s3reader_config)
    enable_sharding = world_size > 1
    logging.info(
        f"Enabled sharding:  {enable_sharding}, because world_size is {world_size}"
    )
    dataset = S3IterableDataset.from_prefix(
        prefix_uri,
        region=region,
        reader_constructor=reader_constructor,
        enable_sharding=enable_sharding,
    )
    dataset = torchdata.datapipes.iter.IterableWrapper(dataset)

    # We don't include when using DDP as this means it's already sharded by iter in S3IterableDataset
    if num_workers > 0 and not dist.is_initialized():
        dataset = dataset.sharding_filter()
    if sharding:
        dataset = dataset.map(tar_to_tuple)
        dataset = dataset.load_from_tar()

    return dataset.map(load_sample)


def create_s3_map_dataset(
    sharding: bool,
    prefix_uri: str,
    region: str,
    load_sample,
    s3reader_config: DictConfig,
    world_size: int = 1,
    rank: int = 0,
):
    reader_constructor = make_s3_reader_constructor(s3reader_config)
    if sharding:
        raise ValueError("Sharding is not supported for s3mapdataset")

    dataset = S3MapDataset.from_prefix(
        prefix_uri,
        region=region,
        transform=load_sample,
        reader_constructor=reader_constructor,
    )
    dataset_size = len(dataset)
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, drop_last=False, shuffle=False
        )
        return dataset, sampler
    return dataset, None


def create_mountpoint_dataset(
    sharding: bool, prefix_uri: str, load_sample, num_workers: int, use_cache: bool
):
    if use_cache:
        cache_dir = tempfile.mkdtemp(dir="./nvme/", prefix="s3mp_cache_")
        arguments = ["--cache", cache_dir, "--metadata-ttl", "indefinite"]
    else:
        arguments = ["--metadata-ttl", "indefinite"]

    prefix_uri = make_mountpoint(prefix_uri=prefix_uri, additional_args=arguments)
    # TODO: compare the performance of using torchdata file APIs and use the more performant option.
    return create_fsspec_dataset(sharding, prefix_uri, load_sample, num_workers)


def create_fsspec_dataset(
    sharding: bool, prefix_uri: str, load_sample, num_workers: int
):
    lister = torchdata.datapipes.iter.FSSpecFileLister(prefix_uri)
    dataset = torchdata.datapipes.iter.FSSpecFileOpener(lister, mode="rb")
    if num_workers > 0:
        dataset = dataset.sharding_filter()
    if sharding:
        dataset = dataset.load_from_tar()

    return dataset.map(load_sample)


def make_dataloader(dataset: Dataset, num_workers: int, batch_size: int, sampler=None):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=default_collate,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=2 if num_workers > 0 else None,
        multiprocessing_context="fork",
    )


# As S3TorchConnector does not implement load_from_tar method
# we are using torchdata by transforming an S3Reader to a tuple of <key, StreamWrapper>
# Since both S3Reader and StreamWrapper are File-Like Objects this transformation is straightforward
def tar_to_tuple(s3object: S3Reader):
    return s3object.key, StreamWrapper(s3object)


if __name__ == "__main__":
    run_experiment()
