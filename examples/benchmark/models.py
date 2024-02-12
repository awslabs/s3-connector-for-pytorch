#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
import abc
import os
import random
import time

import torch
import torch.nn as nn
from PIL import Image
from io import IOBase
from torch.utils.data.dataloader import DataLoader
from omegaconf import DictConfig
from s3torchconnector import S3Reader, S3Checkpoint
from torchvision.transforms import v2
from transformers import ViTForImageClassification

from benchmark_utils import ExperimentResult


class ModelInterface(metaclass=abc.ABCMeta):
    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def load_sample(self, sample: S3Reader | tuple[str, IOBase]):
        """Transform given sample (a file-like Object) to a model's input"""
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, dataloader: DataLoader, epochs: int) -> ExperimentResult:
        """Train the model using given dataloader for number of epochs"""
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, **kwargs):
        """Save checkpoint"""
        raise NotImplementedError


class Entitlement(ModelInterface):
    """
    This is not really a training model as it does not train anything. Instead, this model simply reads the binary
    object data from S3, so that we may identify the max achievable throughput for a given dataset.
    """

    def __init__(self, num_labels: int = None):
        super(Entitlement, self).__init__("Entitlement")
        self.num_labels = num_labels

    def load_sample(self, sample: S3Reader | tuple[str, IOBase]):
        if isinstance(sample, S3Reader):
            key, data = sample.key, sample
        else:
            key, data = sample
        buffer = data.read()
        return len(buffer), key

    def train(self, dataloader: DataLoader, epochs: int) -> ExperimentResult:
        num_samples = 0
        start_time = time.perf_counter()
        for epoch in range(0, epochs):
            for batch_idx, samples in enumerate(dataloader):
                num_samples += len(samples[0])
        end_time = time.perf_counter()
        training_time = end_time - start_time
        return ExperimentResult(training_time=training_time, num_samples=num_samples)

    def save(self):
        raise NotImplementedError


class ViT(ModelInterface):
    """
    Learning Vision Transformer from a pre-trained model.
    See: https://huggingface.co/docs/transformers/model_doc/vit
    """

    def __init__(self, num_labels: int, checkpoint: DictConfig):
        super(ViT, self).__init__("ViT")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = num_labels
        self._model = None
        self._transform = None
        self._optimizer = None
        self.loss_fn = nn.CrossEntropyLoss()
        self.checkpoint = checkpoint

    @property
    def model(self):
        if not self._model:
            model = ViTForImageClassification.from_pretrained(
                "google/vit-base-patch16-224-in21k", num_labels=self.num_labels
            )
            self._model = model.to(self.device)
        return self._model

    @property
    def transform(self):
        if not self._transform:
            self._transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.uint8, scale=True),
                    v2.RandomResizedCrop((224, 224), antialias=True),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        return self._transform

    @property
    def optimizer(self):
        if not self._optimizer:
            self._optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        return self._optimizer

    def load_sample(self, sample: S3Reader | tuple[str, IOBase]):
        if isinstance(sample, S3Reader):
            key, data = sample.key, sample
        else:
            key, data = sample
        img = Image.open(data)
        target = self._get_random_label()
        if self.transform:
            img = self.transform(img)
            return img, target
        return v2.functional.pil_to_tensor(img), target

    # This logic is not specific to Model but actually how we store img-target pairs in S3.
    # As we are not evaluating the model accuracy but actual training time for a
    # fixed number of epochs, it is OK to randomly generate a label between
    # 0 and self.num_labels -1
    def _get_random_label(self):
        return random.randint(0, self.num_labels - 1)

    def train(self, dataloader: DataLoader, epochs: int) -> ExperimentResult:
        num_samples = 0
        start_time = time.perf_counter()
        checkpoint_results = []
        for epoch in range(0, epochs):
            for batch_idx, (data, target) in enumerate(dataloader):
                num_samples += len(data)
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
                    result = self.save(batch_idx=batch_idx + 1)
                    checkpoint_results.append(result)
        end_time = time.perf_counter()
        training_time = end_time - start_time

        return ExperimentResult(
            training_time=training_time,
            num_samples=num_samples,
            checkpoint_results=checkpoint_results,
        )

    def save(self, batch_idx: int):
        destination = self.checkpoint.destination
        if destination == "s3":
            return save_checkpoint_to_s3(
                self.model, self.checkpoint.region, self.checkpoint.uri, batch_idx
            )
        if destination == "disk":
            return save_checkpoint_to_disk(self.model, self.checkpoint.uri, batch_idx)


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
