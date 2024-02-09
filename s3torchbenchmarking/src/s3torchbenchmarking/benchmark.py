#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
import atexit
import shutil
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import Optional, List

import hydra
import torchdata
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, default_collate
from torchdata.datapipes.utils import StreamWrapper

from s3torchbenchmarking.benchmark_utils import ResourceMonitor
from s3torchbenchmarking.models import Entitlement, ViT, ModelInterface
from s3torchconnector import S3IterableDataset, S3Reader, S3MapDataset
from s3torchconnector._s3dataset_common import parse_s3_uri


@hydra.main(version_base=None)
def run_experiment(config: DictConfig):
    model = make_model(config)
    dataset = make_dataset(
        kind=config.dataloader.kind,
        sharding=DatasetSharding.from_conf(config.dataset),
        prefix_uri=config.dataset.prefix_uri,
        region=config.dataset.region,
        load_sample=model.load_sample,
        num_workers=config.dataloader.num_workers,
    )
    dataloader = make_dataloader(
        dataset=dataset,
        num_workers=config.dataloader.num_workers,
        batch_size=config.dataloader.batch_size,
    )

    with ResourceMonitor() as monitor:
        result = model.train(dataloader, config.training.max_epochs)
    result.resource_data = monitor.get_full_data()
    # TODO: Decide if we need to do averaging in Monitor vs in ExperimentResult
    result.avg_resource_data = monitor.get_avg_data()

    # TODO: We are currently only printing the result of the experiment here. We should either write it to a file
    # for further processing, or use CALLBACKS to actually graph/analyse the result. Ideally we should do both.
    print(
        f"{config.dataloader.kind} trained {config.training.model} in "
        f"{result.training_time:.4f}s with {result.throughput:.4f} samples per second"
    )

    print(
        f"Resource usage of the workload was as follows:\n"
        f"{result.avg_resource_data}"
    )


def make_model(config: DictConfig) -> ModelInterface:
    if config.training.model == "entitlement":
        return Entitlement()
    elif config.training.model == "vit":
        num_labels = int(config.training.get("num_labels", 1000))
        return ViT(num_labels, config.checkpoint)
    else:
        raise Exception(f"unknown model {config.training.model}")


class DatasetSharding(Enum):
    TAR = 1

    @staticmethod
    def from_conf(dataset_config: DictConfig):
        if dataset_config.get("sharding"):
            return DatasetSharding[dataset_config.sharding]


def make_mountpoint(
    prefix_uri: str,
    mountpoint_path: Optional[str] = None,
    additional_args: List[str] = None,
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
    kind: str,
    sharding: Optional[DatasetSharding],
    prefix_uri: str,
    region: str,
    load_sample,
    num_workers: int,
):
    if kind == "s3iterabledataset":
        return create_s3_iterable_dataset(
            sharding, prefix_uri, region, load_sample, num_workers
        )
    elif kind == "s3mapdataset":
        return create_s3_map_dataset(sharding, prefix_uri, region, load_sample)
    elif kind == "fsspec":
        return create_fsspec_dataset(sharding, prefix_uri, load_sample, num_workers)
    elif kind == "mountpoint":
        return create_mountpoint_dataset(sharding, prefix_uri, load_sample, num_workers)
    else:
        raise Exception(f"unknown dataset kind {kind}")


def create_s3_iterable_dataset(
    sharding: Optional[DatasetSharding],
    prefix_uri: str,
    region: str,
    load_sample,
    num_workers: int,
):
    dataset = S3IterableDataset.from_prefix(prefix_uri, region=region)
    dataset = torchdata.datapipes.iter.IterableWrapper(dataset)
    if num_workers > 0:
        dataset = dataset.sharding_filter()
    if sharding == DatasetSharding.TAR:
        dataset = dataset.map(tar_to_tuple)
        dataset = dataset.load_from_tar()

    return dataset.map(load_sample)


def create_s3_map_dataset(
    sharding: Optional[DatasetSharding], prefix_uri: str, region: str, load_sample
):
    if sharding:
        raise ValueError("Sharding is not supported for s3mapdataset")
    else:
        dataset = S3MapDataset.from_prefix(
            prefix_uri, region=region, transform=load_sample
        )
    return dataset


def create_mountpoint_dataset(
    sharding: Optional[DatasetSharding], prefix_uri: str, load_sample, num_workers: int
):
    prefix_uri = make_mountpoint(prefix_uri=prefix_uri)
    # TODO: compare the performance of using torchdata file APIs and use the more performant option.
    return create_fsspec_dataset(sharding, prefix_uri, load_sample, num_workers)


def create_fsspec_dataset(
    sharding: Optional[DatasetSharding],
    prefix_uri: str,
    load_sample: object,
    num_workers: int,
):
    lister = torchdata.datapipes.iter.FSSpecFileLister(prefix_uri)
    dataset = torchdata.datapipes.iter.FSSpecFileOpener(lister, mode="rb")
    if num_workers > 0:
        dataset = dataset.sharding_filter()
    if sharding == DatasetSharding.TAR:
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
