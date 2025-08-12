#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
import atexit
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List

import hydra
import torchdata  # type: ignore
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, default_collate
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

# TODO: add Structured Config (https://hydra.cc/docs/tutorials/structured_config/intro/)
@hydra.main(version_base=None)
def run_experiment(config: DictConfig) -> dict:
    model = make_model(config)

    fully_qualified_uri = (
        "s3://" + config.s3.bucket.strip("/") + "/" + config.dataset.strip("/")
    )
    num_of_gpus = torch.cuda.device_count()
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    logging.info(f"CUDA version: {torch.version.cuda}")
    logging.info(f"GPU count: {torch.cuda.device_count()}")

    logging.info(f"Using {num_of_gpus} GPUs")
    dataset = make_dataset(
        dataloader_config=config.dataloader,
        sharding=config.sharding,
        prefix_uri=fully_qualified_uri,
        region=config.s3.region,
        load_sample=model.load_sample,
    )
    dataloader = make_dataloader(
        dataset=dataset,
        num_workers=config.dataloader.num_workers,
        batch_size=config.dataloader.batch_size,
    )

    result: ExperimentResult = model.train(dataloader, config.epochs)

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
        )
    if kind == "s3mapdataset":
        if not region:
            raise ValueError("Must provide region for s3mapdataset")
        if not dataloader_config.get("s3reader"):
            raise ValueError(f"Must provide s3reader config for {kind}")
        s3reader_config = dataloader_config.s3reader
        return create_s3_map_dataset(
            sharding, prefix_uri, region, load_sample, s3reader_config
        )
    if kind == "fsspec":
        return create_fsspec_dataset(sharding, prefix_uri, load_sample, num_workers)
    if kind == "mountpoint":
        return create_mountpoint_dataset(
            sharding, prefix_uri, load_sample, num_workers, False
        )
    if kind == "mountpointcache":
        return create_mountpoint_dataset(
            sharding, prefix_uri, load_sample, num_workers, True
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
):
    reader_constructor = make_s3_reader_constructor(s3reader_config)
    dataset = S3IterableDataset.from_prefix(
        prefix_uri, region=region, reader_constructor=reader_constructor
    )
    dataset = torchdata.datapipes.iter.IterableWrapper(dataset)

    if num_workers > 0:
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
):
    reader_constructor = make_s3_reader_constructor(s3reader_config)
    if sharding:
        raise ValueError("Sharding is not supported for s3mapdataset")
    else:
        dataset = S3MapDataset.from_prefix(
            prefix_uri,
            region=region,
            transform=load_sample,
            reader_constructor=reader_constructor,
        )
    return dataset


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


def make_dataloader(dataset: Dataset, num_workers: int, batch_size: int):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
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


if __name__ == "__main__":
    run_experiment()
