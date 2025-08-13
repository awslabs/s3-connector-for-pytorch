#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
import atexit
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List
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
logger = logging.getLogger(__name__)

def init_distributed(config: DictConfig):
    if torch.cuda.device_count() > 1:
        if not dist.is_initialized():
            # torchrun sets these environment variables automatically
            dist.init_process_group(backend="nccl")
            
            # Set device for this process
            local_rank = int(os.environ.get('LOCAL_RANK', '0'))
            torch.cuda.set_device(local_rank)
            
            logging.info(f"Initialized DDP: rank {dist.get_rank()}/{dist.get_world_size()} on GPU {local_rank}")
        atexit.register(cleanup_distributed)


        
def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()
        logging.info("Cleaned up distributed training")
        
        
# TODO: add Structured Config (https://hydra.cc/docs/tutorials/structured_config/intro/)
@hydra.main(version_base=None)
def run_experiment(config: DictConfig) -> dict:
    init_distributed(config)
    
    num_gpus = torch.cuda.device_count()
    world_size = dist.get_world_size() if dist.is_initialized else 1
    rank = dist.get_rank() if dist.is_initialized else 0
    
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    logging.info(f"CUDA version: {torch.version.cuda}")
    logging.info(f"GPU count: {num_gpus}")
    logging.info(f"World size: {world_size}")
    logging.info(f"Rank: {rank}")
    
        # Validation and warnings
    if num_gpus > 1 and world_size == 1:
        logging.warning(f"Multiple GPUs detected ({num_gpus}) but running in single-GPU mode. Consider using torchrun for distributed training.")
    elif num_gpus > 1 and world_size > 1:
        logging.info(f"Multi-GPU distributed training: {world_size} processes across {num_gpus} GPUs")
    elif num_gpus <= 1:
        logging.info("Single GPU or CPU-only training")
    

    model = make_model(config)
 
 
    fully_qualified_uri = (
        "s3://" + config.s3.bucket.strip("/") + "/" + config.dataset.strip("/")
    )
    dataset, sampler = make_dataset(
        dataloader_config=config.dataloader,
        sharding=config.sharding,
        prefix_uri=fully_qualified_uri,
        region=config.s3.region,
        load_sample=model.load_sample,
    )
    
    validate_dataset_coverage(dataset, sampler, world_size, rank)

    dataloader = make_dataloader(
        dataset=dataset,
        sampler=sampler,
        num_workers=config.dataloader.num_workers,
        batch_size=config.dataloader.batch_size,
    )

    result: ExperimentResult = model.train(dataloader, config.epochs)
    
    # Only return metrics from rank 0 to avoid duplicates
    if rank != 0:
        return {}
    
    metrics = {
        "throughput_mibs": result["volume"] / result["training_duration_s"],
        "training_duration_s": result["training_duration_s"],
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
) -> Dataset:
    world_size = dist.get_world_size() if dist.is_initialized else 1
    rank = dist.get_rank() if dist.is_initialized else 0
    kind = dataloader_config.kind
    num_workers = dataloader_config.num_workers

    if kind == "s3iterabledataset":
        if not region:
            raise ValueError("Must provide region for s3iterabledataset")
        if not dataloader_config.get("s3reader"):
            raise ValueError(f"Must provide s3reader config for {kind}")
        s3reader_config = dataloader_config.s3reader
        return create_s3_iterable_dataset(
            sharding,
            prefix_uri,
            region,
            load_sample,
            num_workers,
            s3reader_config,
            world_size,
            rank,
        ), None
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
        return create_fsspec_dataset(sharding, prefix_uri, load_sample, num_workers), None
    if kind == "mountpoint":
        return create_mountpoint_dataset(
            sharding, prefix_uri, load_sample, num_workers, False
        ), None
    if kind == "mountpointcache":
        return create_mountpoint_dataset(
            sharding, prefix_uri, load_sample, num_workers, True
        ), None
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
    rank: int = 0,
):
    reader_constructor = make_s3_reader_constructor(s3reader_config)
    enable_sharding = world_size > 1
    dataset = S3IterableDataset.from_prefix(
        prefix_uri, region=region, reader_constructor=reader_constructor, enable_sharding= enable_sharding)
=    validate_s3_iterable_dataset_sharding(dataset, world_size, rank, num_workers)

    dataset = torchdata.datapipes.iter.IterableWrapper(dataset)

    if num_workers > 0:
        dataset = dataset.sharding_filter()
    if sharding:
        dataset = dataset.map(tar_to_tuple)
        dataset = dataset.load_from_tar()

    return dataset.map(load_sample)

def validate_s3_iterable_dataset_sharding(dataset, world_size, rank, num_workers):
    """Validate S3IterableDataset sharding by sampling the first few items"""
    if world_size <= 1 and num_workers <= 1:
        logging.info("Single process, single worker - no sharding validation needed")
        return
    
    # Sample first 20 items to check sharding
    sample_keys = []
    sample_count = 0    
    try:
        for item in dataset:
            if hasattr(item, 'key'):
                sample_keys.append(item.key)
            elif isinstance(item, tuple) and len(item) >= 2:
                sample_keys.append(str(item[1]))  # key from load_sample
            else:
                sample_keys.append(f"item_{sample_count}")
            
            sample_count += 1
 
    except Exception as e:
        logging.warning(f"Could not sample dataset for validation: {e}")
        return
    
    logging.info(f"Rank {rank}: Sampled {sample_count} items from S3IterableDataset")
    logging.info(f"Rank {rank}: Sample keys: {sample_keys[:5]}...")  # Show first 5
    
    if world_size > 1:
        logging.info(f"Rank {rank}: S3IterableDataset sharding enabled - each rank should see different objects")
    if num_workers > 0:
        logging.info(f"Rank {rank}: Worker sharding enabled with {num_workers} workers")

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
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        logger.info(f"Using DistributedSampler for S3MapDataset with {world_size} processes")
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


def make_dataloader(dataset: Dataset, num_workers: int, batch_size: int, sampler = None):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler = sampler,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=default_collate,
    )


# As S3TorchConnector does not implement load_from_tar method
# we are using torchdata by transforming an S3Reader to a tuple of <key, StreamWrapper>
# Since both S3Reader and StreamWrapper are File-Like Objects this transformation is straightforward
def tar_to_tuple(s3object: S3Reader):
    return s3object.key, StreamWrapper(s3object)

def validate_dataset_coverage(dataset, sampler, world_size, rank):
    """Validate that the entire dataset is covered across all processes without duplication"""
    if world_size <= 1:
        logging.info("Single process - no dataset coverage validation needed")
        return
    
    # For S3MapDataset with DistributedSampler
    if hasattr(dataset, '__len__') and sampler is not None:
        total_samples = len(dataset)
        sampler_indices = list(sampler)
        expected_samples_per_rank = total_samples // world_size
        
        logging.info(f"Rank {rank}: Dataset size={total_samples}, Sampler indices={len(sampler_indices)}")
        logging.info(f"Rank {rank}: Expected samples per rank: {expected_samples_per_rank}")
        
        if len(sampler_indices) != expected_samples_per_rank:
            logging.warning(f"Rank {rank}: Expected {expected_samples_per_rank} samples, got {len(sampler_indices)}")
        else:
            logging.info(f"Rank {rank}: Dataset coverage validation passed")
    
    # For S3IterableDataset (built-in sharding)
    elif hasattr(dataset, '_enable_sharding'):
        logging.info(f"Rank {rank}: S3IterableDataset with built-in sharding - coverage handled internally")
    
    else:
        logging.info(f"Rank {rank}: Dataset type doesn't support coverage validation")


if __name__ == "__main__":
    run_experiment()
