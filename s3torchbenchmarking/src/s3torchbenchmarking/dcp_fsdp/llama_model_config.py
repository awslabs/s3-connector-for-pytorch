#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from dataclasses import dataclass
from transformers import LlamaConfig, AutoModelForCausalLM


@dataclass
class LlamaModelConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int


# LlamaModelParams is a class that takes a model size as input and returns the corresponding model configuration
class LlamaModelParams:
    def __init__(self, model_size: str):
        configs = {
            "L7b": LlamaModelConfig(4096, 11008, 32, 32, 32),
            "L13b": LlamaModelConfig(5120, 13824, 40, 40, 40),
            # "mixed": LlamaModelConfig(8192, 20000, 40, 64, 8),  # 101194
            "L30b": LlamaModelConfig(6656, 17920, 60, 52, 52),  # 125024
            "L65b": LlamaModelConfig(8192, 22016, 80, 64, 64),
            "L70b": LlamaModelConfig(8192, 28672, 80, 64, 8),
        }

        if model_size not in configs:
            raise ValueError(f"Invalid model size. Choose from: {list(configs.keys())}")

        config = configs[model_size]
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads


# create a function that returns a llama model config
def get_llama_model_config(model_name: str):
    params = LlamaModelParams(model_name)
    model_config = LlamaConfig(
        vocab_size=50432,
        hidden_size=params.hidden_size,
        intermediate_size=params.intermediate_size,
        num_hidden_layers=params.num_hidden_layers,
        num_attention_heads=params.num_attention_heads,
        num_key_value_heads=params.num_key_value_heads,
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        use_cache=False,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_scaling=None,
    )
    return model_config


# create a function that returns a llama model
def get_llama_model(model_name: str):
    model_config = get_llama_model_config(model_name)
    model = AutoModelForCausalLM.from_config(model_config)
    return model
