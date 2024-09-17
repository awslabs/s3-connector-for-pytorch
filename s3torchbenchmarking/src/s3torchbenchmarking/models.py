#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
import abc
import os
import random
import time
from functools import cached_property
from io import IOBase
from typing import Optional, Any, Tuple, Union

import lightning as L
import torch
import torch.nn as nn
import torchdata  # type: ignore
from PIL import Image
from lightning.pytorch import callbacks
from lightning.pytorch.strategies import SingleDeviceStrategy
from omegaconf import DictConfig
from s3torchconnector import S3Reader, S3Checkpoint, S3IterableDataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import v2  # type: ignore
from transformers import ViTForImageClassification  # type: ignore

from .benchmark_utils import (
    ExperimentResult,
    ResourceMonitor,
    Distribution,
    Transforms,
)
from .lightning_utils.checkpoint_profiler import CheckpointProfiler
from .lightning_utils.sample_counter import SampleCounter
from s3torchconnector.lightning import S3LightningCheckpoint  # type: ignore


class ModelInterface(metaclass=abc.ABCMeta):
    def __init__(self):
        self.name = self.__class__.__name__

    def load_sample(self, sample: Union[S3Reader, Tuple[str, IOBase]]):
        """Transform given sample (a file-like Object) to a model's input"""
        if isinstance(sample, S3Reader):
            key, data = sample.key, sample
        else:
            key, data = sample

        return key, data

    @abc.abstractmethod
    def train_batch(self, batch_idx: int, data, target) -> Optional[Any]:
        raise NotImplementedError

    def train(self, dataloader: DataLoader, epochs: int) -> ExperimentResult:
        """Train the model using given dataloader for number of epochs"""
        with ResourceMonitor() as monitor:
            num_samples = 0
            start_time = time.perf_counter()
            checkpoint_times = Distribution(initial_capacity=1024)
            for epoch in range(0, epochs):
                for batch_idx, (data, target) in enumerate(dataloader):
                    result = self.train_batch(batch_idx, data, target)
                    num_samples += len(data)
                    if result:
                        checkpoint_times.add(result)
            end_time = time.perf_counter()
            training_time = end_time - start_time

        return ExperimentResult(
            elapsed_time=training_time,
            volume=num_samples,
            checkpoint_times=checkpoint_times,
            utilization=monitor.resource_data,
        )

    @abc.abstractmethod
    def save(self, **kwargs):
        """Save checkpoint"""
        raise NotImplementedError


class Entitlement(ModelInterface):
    """
    This is not really a training model as it does not train anything. Instead, this model simply reads the binary
    object data from S3, so that we may identify the max achievable throughput for a given dataset.
    """

    def __init__(self, num_labels: Optional[int] = None):
        super().__init__()
        self.num_labels = num_labels

    def load_sample(self, sample: Union[S3Reader, Tuple[str, IOBase]]):
        key, data = super().load_sample(sample)
        buffer = data.read()
        return len(buffer), key

    def train_batch(self, batch_idx: int, data, target):
        pass

    def save(self):
        raise NotImplementedError


class ViT(ModelInterface):
    """
    Learning Vision Transformer from a pre-trained model.
    See: https://huggingface.co/docs/transformers/model_doc/vit
    """

    def __init__(self, num_labels: int, checkpoint: DictConfig):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = num_labels
        self.loss_fn = nn.CrossEntropyLoss()
        self.checkpoint = checkpoint

    @cached_property
    def model(self):
        return ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k", num_labels=self.num_labels
        ).to(self.device)

    @cached_property
    def transform(self):
        return v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True),
                v2.RandomResizedCrop((224, 224), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @cached_property
    def optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)

    def load_sample(self, sample: Union[S3Reader, Tuple[str, IOBase]]):
        key, data = super().load_sample(sample)
        img = Image.open(data)
        target = self._get_random_label()
        if self.transform:
            img = self.transform(img)
            return img, target
        return v2.functional.pil_to_tensor(img), target

    # This logic is not specific to Model but actually how we store img-target pairs in S3.
    # As we are not evaluating the model accuracy but actual training time for a
    # fixed number of epochs, it is OK to randomly generate a label between
    # 0 and self.num_labels - 1.
    def _get_random_label(self):
        return random.randint(0, self.num_labels - 1)

    def train_batch(self, batch_idx: int, data, target) -> Optional[float]:
        data = data.to(self.device)
        target = target.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(data)
        loss = self.loss_fn(outputs.logits, target)
        loss.backward()
        self.optimizer.step()
        if (
            self.checkpoint.save_one_in > 0
            and (batch_idx + 1) % self.checkpoint.save_one_in == 0
        ):
            return self.save(batch_idx=batch_idx + 1)
        return None

    def save(self, batch_idx: int):
        destination = self.checkpoint.destination
        if destination == "s3":
            return save_checkpoint_to_s3(
                self.model, self.checkpoint.region, self.checkpoint.uri, batch_idx
            )
        if destination == "disk":
            return save_checkpoint_to_disk(self.model, self.checkpoint.uri, batch_idx)


class LightningAdapter(ModelInterface):
    class DelegateModule(L.LightningModule):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.loss_fn = nn.CrossEntropyLoss()

        def forward(self, inputs, target):
            return self.model(inputs, target)

        def training_step(self, batch, batch_idx):
            inputs, target = batch
            output = self(inputs, target)
            loss = self.loss_fn(output.logits, target)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.model.parameters(), lr=0.001)

    def __init__(self, model, sample_transformer, config: DictConfig):
        super().__init__()
        self.config = config
        self.sample_transformer = sample_transformer
        self.lightning_model = LightningAdapter.DelegateModule(model)

    def load_sample(self, sample: Union[S3Reader, Tuple[str, IOBase]]):
        _, data = super().load_sample(sample)

        return self.sample_transformer(data), self._get_random_label()

    def _get_random_label(self):
        return random.randrange(1024)

    def train_batch(self, batch_idx: int, data, target) -> Optional[Any]:
        pass

    def train(self, dataloader: DataLoader, epochs: int) -> ExperimentResult:
        strategy = SingleDeviceStrategy()
        checkpoint_io = strategy.checkpoint_io
        sample_counting_cb = SampleCounter()
        checkpoint_callback = callbacks.ModelCheckpoint(
            dirpath=self.config.checkpoint.uri,
            every_n_train_steps=self.config.checkpoint.save_one_in,
            save_on_train_epoch_end=True,
        )
        checkpoint_dest = self.config.checkpoint.destination
        if checkpoint_dest == "s3":
            checkpoint_io = S3LightningCheckpoint(self.config.checkpoint.region)
        profiling_checkpointer = CheckpointProfiler(checkpoint_io)
        trainer = L.Trainer(
            # log_every_n_steps=10,
            logger=None,
            # limit_train_batches=10,
            plugins=[profiling_checkpointer],
            max_epochs=epochs,
            # profiler="simple",
            callbacks=[
                checkpoint_callback,
                sample_counting_cb,
                # callbacks.device_stats_monitor.DeviceStatsMonitor(cpu_stats=True)
            ],
        )

        with ResourceMonitor() as monitor:
            start_time = time.perf_counter()
            trainer.fit(model=self.lightning_model, train_dataloaders=dataloader)
            end_time = time.perf_counter()
            training_time = end_time - start_time

        return ExperimentResult(
            elapsed_time=training_time,
            volume=sample_counting_cb.count,
            checkpoint_times=profiling_checkpointer.save_times,
            utilization=monitor.resource_data,
        )

    def save(self, **kwargs):
        raise NotImplementedError(
            "Checkpoint functionality is built into the Lightning trainer."
        )


def save_checkpoint_to_s3(model: nn.Module, region: str, uri: str, batch_idx: int):
    checkpoint = S3Checkpoint(region=region)
    # Save checkpoint to S3
    start_time = time.perf_counter()
    with checkpoint.writer(uri + f"batch{batch_idx}.ckpt") as writer:
        torch.save(model.state_dict(), writer)
    end_time = time.perf_counter()
    save_time = end_time - start_time
    print(f"Saving checkpoint to {uri} took {save_time} seconds")
    return save_time


def save_checkpoint_to_disk(model: nn.Module, uri: str, batch_idx: int):
    if not os.path.exists(uri):
        os.makedirs(uri)
    path = os.path.join(uri, f"batch{batch_idx}.ckpt")
    start_time = time.perf_counter()
    torch.save(model.state_dict(), path)
    end_time = time.perf_counter()
    save_time = end_time - start_time
    print(f"Saving checkpoint to {path} took {save_time} seconds")
    return save_time
