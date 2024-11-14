#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from functools import cached_property
from typing import Callable

from torch.nn import Module
from transformers import AutoModelForSeq2SeqLM, ViTModel, CLIPModel


class BenchmarkModel:
    """Utility class around a :class:`torch.nn.Module`, with an additional metadata layer."""

    def __init__(self, loader: Callable, name: str):
        self._loader = loader
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @cached_property
    def model(self) -> Module:
        return self._loader(self._name)

    @cached_property
    def size(self) -> float:
        """Compute a model's size (in MiB).

        Sourced from https://discuss.pytorch.org/t/finding-model-size/130275/2.
        """
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return (param_size + buffer_size) / 1024**2


# NOTE: keys below are later used to construct a filename, so make sure they do not contain characters that will not
# play well with filesystems (e.g., '/').
SIZE_TO_MODEL = {
    # ~350 MB model
    "vit-base": BenchmarkModel(
        ViTModel.from_pretrained, "google/vit-base-patch16-224-in21k"
    ),
    # ~1.7 GB model
    "clip-vit": BenchmarkModel(
        CLIPModel.from_pretrained, "openai/clip-vit-large-patch14"
    ),
    # ~12 GB model
    "T0_3B": BenchmarkModel(AutoModelForSeq2SeqLM.from_pretrained, "bigscience/T0_3B"),
    # ~45 GB model
    "T0pp": BenchmarkModel(AutoModelForSeq2SeqLM.from_pretrained, "bigscience/T0pp"),
}


def get_benchmark_model(name: str) -> BenchmarkModel:
    """Select a model for benchmarking."""
    if name not in SIZE_TO_MODEL:
        raise ValueError(f'Name "{name}" is unexpected')
    return SIZE_TO_MODEL[name]
