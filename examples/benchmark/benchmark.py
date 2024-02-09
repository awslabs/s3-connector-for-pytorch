#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
import hydra
import torchdata
from omegaconf import DictConfig
from s3torchconnector import S3IterableDataset, S3Reader, S3MapDataset
from torch.utils.data import DataLoader, default_collate
from torchdata.datapipes.utils import StreamWrapper

from benchmark_utils import ResourceMonitor
from models import Entitlement, ViT


@hydra.main(version_base=None, config_path="configuration")
def run_experiment(config: DictConfig):
    if config.training.model == "entitlement":
        model = Entitlement()
    elif config.training.model == "vit":
        model = ViT(1000, config.checkpoint)
    else:
        raise Exception(f"unknown model {config.training.model}")

    dataset = make_dataset(
        config.dataloader.kind,
        config.dataset.sharding,
        config.dataset.prefix_uri,
        config.dataset.region,
        model.load_sample,
        num_workers=config.dataloader.num_workers,
    )
    dataloader = make_dataloader(
        dataset, config.dataloader.num_workers, config.dataloader.batch_size
    )

    monitor = ResourceMonitor()
    monitor.start()
    result = model.train(dataloader, config.training.max_epochs)
    result.resource_data = monitor.get_full_data()

    # TODO: Decide if we need to do averaging in Monitor vs in ExperimentResult
    result.avg_resource_data = monitor.get_avg_data()
    monitor.stop()

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


def make_dataset(
    kind: str,
    sharded: bool,
    prefix_uri: str,
    region: str,
    load_sample,
    num_workers: int,
):
    if kind == "s3iterabledataset":
        dataset = S3IterableDataset.from_prefix(prefix_uri, region=region)
        dataset = torchdata.datapipes.iter.IterableWrapper(dataset)
        if num_workers > 0:
            dataset = dataset.sharding_filter()
        if sharded:
            dataset = dataset.map(tar_to_tuple)
            dataset = dataset.load_from_tar()
        return dataset.map(load_sample)
    if kind == "s3mapdataset":
        if sharded:
            raise Exception(f"Sharding is not supported for {kind}")
        else:
            dataset = S3MapDataset.from_prefix(
                prefix_uri, region=region, transform=load_sample
            )
            return dataset
    if kind == "fsspec":
        lister = torchdata.datapipes.iter.FSSpecFileLister(prefix_uri)
        dataset = torchdata.datapipes.iter.FSSpecFileOpener(lister, mode="rb")
        if num_workers > 0:
            dataset = dataset.sharding_filter()
        if sharded:
            dataset = dataset.load_from_tar()
        return dataset.map(load_sample)
    else:
        raise Exception(f"unknown dataset kind {kind}")


def make_dataloader(dataset, num_workers: int, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
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
