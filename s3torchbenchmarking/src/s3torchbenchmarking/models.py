#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from functools import cached_property
from io import IOBase
from time import perf_counter
from typing import Optional, Any, Tuple, Union, Callable, List

import lightning as L
import torch
import torch.nn as nn
from PIL import Image
from lightning.pytorch import callbacks
from lightning.pytorch.strategies import SingleDeviceStrategy
from omegaconf import DictConfig
from torch.nn import Module
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import v2  # type: ignore
import itertools
from transformers import (  # type: ignore
    ViTForImageClassification,
    ViTModel,
    CLIPModel,
    AutoModelForSeq2SeqLM,
    AutoModelForSpeechSeq2Seq,
    AutoModelForCausalLM,
)
import torch.distributed as dist

from s3torchconnector import S3Reader, S3Checkpoint
from s3torchconnector.lightning import S3LightningCheckpoint
from .benchmark_utils import ExperimentResult, ResourceMonitor
from .dcp_fsdp.llama_model_config import get_llama_model
from .lightning_checkpointing.checkpoint_profiler import CheckpointProfiler
from .lightning_checkpointing.sample_counter import SampleCounter

logger = logging.getLogger(__name__)


class BenchmarkModel:
    """Utility class around a :class:`torch.nn.Module`, with an additional metadata layer.

    Args:
        loader (Callable): Function to load a pretrained Hugging Face model.
        name (str): Name of the pretrained Hugging Face model.
        **kwargs: Additional keyword arguments to pass to the :class:`torch.nn.Module`.
    """

    def __init__(self, loader: Callable, name: str, **kwargs):
        self._loader = loader
        self._name = name
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        return self._name

    @cached_property
    def model(self) -> Module:
        """Instantiate the pretrained model."""
        model_load_start = perf_counter()
        model = self._loader(self._name, **self._kwargs)
        model_load_end = perf_counter()
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            logger.info(
                f"Rank:{rank}. Time taken to load model {self._name}: {model_load_end - model_load_start} seconds"
            )
        return model

    @cached_property
    def size(self) -> float:
        """Compute a model's size (in MiB).

        Note:
            Sourced from https://discuss.pytorch.org/t/finding-model-size/130275/2.
        """
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return (param_size + buffer_size) / 1024**2


_MODELS = {
    # ~350 MB model
    "vit-base": BenchmarkModel(
        ViTModel.from_pretrained, "google/vit-base-patch16-224-in21k"
    ),
    # ~1.7 GB model
    "clip-vit": BenchmarkModel(
        CLIPModel.from_pretrained, "openai/clip-vit-large-patch14"
    ),
    # ?
    "whisper": BenchmarkModel(
        AutoModelForSpeechSeq2Seq.from_pretrained,
        "openai/whisper-large-v3",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ),
    # ~12 GB model
    "T0_3B": BenchmarkModel(AutoModelForSeq2SeqLM.from_pretrained, "bigscience/T0_3B"),
    # ~45 GB model
    "T0pp": BenchmarkModel(AutoModelForSeq2SeqLM.from_pretrained, "bigscience/T0pp"),
    # ~25.6 GB model
    "L7b": BenchmarkModel(get_llama_model, "L7b"),
    # ~
    "L13b": BenchmarkModel(get_llama_model, "L13b"),
    # ~125 GB model
    "L30b": BenchmarkModel(get_llama_model, "L30b"),
    # ~
    "L65b": BenchmarkModel(get_llama_model, "L65b"),
    # ~
    "L70b": BenchmarkModel(get_llama_model, "L70b"),
}


def get_benchmark_model(short_name: str) -> BenchmarkModel:
    """Select a model for benchmarking."""
    return _MODELS[short_name]


class ModelInterface(ABC):
    """Abstract interface for model interface."""

    def load_sample(self, sample: Union[S3Reader, Tuple[str, IOBase]]):
        """Transform given sample (a file-like Object) to a model's input"""
        if isinstance(sample, S3Reader):
            key, data = sample.key, sample
        else:
            key, data = sample

        return key, data

    @abstractmethod
    def train_batch(self, batch_idx: int, data, target) -> Optional[Any]:
        raise NotImplementedError

    def capped_loader(self, loader):
        """Cap the number of steps in the loader to the minimum number of steps across all ranks"""
        if not dist.is_initialized():
            yield from loader
            return
        world = dist.get_world_size()
        
        try:
            local_steps = len(loader)        # in the case of map style datasets we can use len as we know the size of the loader
        except TypeError:
            local_steps = None
            
        if local_steps is not None:
            counts = [None] * world
            dist.all_gather_object(counts, local_steps)
            min_steps = min(counts)
            yield from itertools.islice(loader, min_steps) 
            return
        
        # In the case of iterable datasets we need to to use iter
        it = iter(loader)
        # Use cuda with nccl if available for the purpose of using dist.all_reduce
        dev = torch.device("cuda", torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")) 
        
        # We use torch tensors as it's faster to reduce and serialize
        flag = torch.zeros(1, device = dev, dtype = torch.int32)
        while True:
            # Pull out one batch one at at time, if it works on all ranks then we yield else stop
            try:
                batch = next(it)
                flag.fill_(1)
            except:
                batch = None
                flag.zero_()
            # We know there's multiple gpus in this case
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            if int(flag.item()) == 1:
                yield batch
            else:
                break


    def train(self, dataloader: DataLoader, epochs: int) -> ExperimentResult:
        """Train the model using given dataloader for number of epochs"""

        epoch_durations_s: List[float] = []

        with ResourceMonitor() as monitor:
            num_samples = 0
            checkpoint_times = []
            begin_training = perf_counter()
            for epoch in range(epochs):
                begin_epoch = time.perf_counter()
                logger.info("Epoch #%i/%i", epoch, epochs - 1)
                batch_count = 0
                try:
                    for batch_idx, (data, target) in enumerate(
                        self.capped_loader(dataloader)
                    ):
                        logger.debug("Batch #%i", batch_idx)
                        result = self.train_batch(batch_idx, data, target)
                        num_samples += len(data)
                        batch_count += 1
                        if batch_count % 1000 == 0:
                            logger.info(
                                f"Processed {batch_count} batches, {num_samples} samples"
                            )
                        if result:
                            checkpoint_times.append(result)
                except Exception as e:
                    logger.error(f"Error in training loop at batch {batch_count}: {e}")
                    raise
                logger.info(f"Epoch {epoch} completed with {batch_count} batches")
                epoch_durations_s.append(time.perf_counter() - begin_epoch)
            training_duration_s = time.perf_counter() - begin_training

        return {
            "training_duration_s": training_duration_s,
            "epoch_durations_s": epoch_durations_s,
            "volume": num_samples,
            "checkpoint_times": checkpoint_times,
            "utilization": monitor.resource_data,
        }

    @abstractmethod
    def save(self, **kwargs):
        """Save checkpoint"""
        raise NotImplementedError


class Entitlement(ModelInterface):
    """
    This is not really a training model as it does not train anything. Instead, this model simply reads the binary
    object data from S3, so that we may identify the max achievable throughput for a given dataset.
    """

    def __init__(self, num_labels: Optional[int] = None):
        self.num_labels = num_labels
        self.model = None

    def load_sample(self, sample: Union[S3Reader, Tuple[str, IOBase]]):
        key, data = super().load_sample(sample)
        try:
            buffer = data.read()
            return len(buffer), key
        except Exception as e:
            logger.warning(f"Failed to read sample {key}: {e}")
            return 0, key

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

        return {
            "training_duration_s": training_time,
            "volume": sample_counting_cb.count,
            "checkpoint_times": profiling_checkpointer.save_times,
            "utilization": monitor.resource_data,
        }

    def save(self, **kwargs):
        raise NotImplementedError(
            "Checkpoint functionality is built into the Lightning trainer."
        )


def save_checkpoint_to_s3(model: Module, region: str, uri: str, batch_idx: int):
    checkpoint = S3Checkpoint(region=region)
    start_time = time.perf_counter()
    with checkpoint.writer(uri + f"batch{batch_idx}.ckpt") as writer:
        torch.save(model.state_dict(), writer)
    end_time = time.perf_counter()
    save_time = end_time - start_time
    print(f"Saving checkpoint to {uri} took {save_time} seconds")
    return save_time


def save_checkpoint_to_disk(model: Module, uri: str, batch_idx: int):
    if not os.path.exists(uri):
        os.makedirs(uri)
    path = os.path.join(uri, f"batch{batch_idx}.ckpt")
    start_time = time.perf_counter()
    torch.save(model.state_dict(), path)
    end_time = time.perf_counter()
    save_time = end_time - start_time
    print(f"Saving checkpoint to {path} took {save_time} seconds")
    return save_time
