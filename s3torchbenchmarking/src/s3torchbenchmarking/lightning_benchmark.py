import json
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch import callbacks
from lightning.pytorch.strategies import SingleDeviceStrategy
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper  # type: ignore
from transformers import CLIPModel, AutoModelForSeq2SeqLM  # type: ignore
from transformers import ViTForImageClassification, AutoModelForSpeechSeq2Seq

from .benchmark_utils import ResourceMonitor, Distribution
from .lightning_utils.checkpoint_profiler import CheckpointProfiler
from .models import LightningAdapter
from s3torchconnector._s3dataset_common import parse_s3_uri  # type: ignore
from s3torchconnector.lightning import S3LightningCheckpoint  # type: ignore


def run_lightning_experiment(config: DictConfig):
    if config.training.model == "vit":
        # 330MB model
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k", num_labels=1024
        )
    elif config.training.model == "whisper":
        # 2.9GB model
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            "openai/whisper-large-v3",
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
    elif config.training.model == "clip":
        # 1.6GB model
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    elif config.training.model == "t0_3b":
        # 10GB model
        model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0_3B")
    elif config.training.model == "t0pp":
        # 40GB model
        model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp")
    else:
        raise NotImplementedError(
            f"The model: {config.training.model} is currently not supported"
        )

    strategy = SingleDeviceStrategy()
    checkpoint_io = strategy.checkpoint_io
    checkpoint_callback = callbacks.ModelCheckpoint(dirpath=config.checkpoint.uri)
    checkpoint_dest = config.checkpoint.destination
    if checkpoint_dest == "s3":
        checkpoint_io = S3LightningCheckpoint(config.checkpoint.region)
    profiling_checkpointer = CheckpointProfiler(checkpoint_io)
    trainer = L.Trainer(
        logger=None,
        plugins=[profiling_checkpointer],
        callbacks=[
            checkpoint_callback,
        ],
    )

    dataloader = DataLoader(IterableWrapper([]), num_workers=8)
    trainer.fit(LightningAdapter.DelegateModule(model), train_dataloaders=dataloader)
    with ResourceMonitor() as monitor:
        for i in range(config.training.max_epochs):
            checkpoint_path = build_checkpoint_path(
                config.checkpoint.uri, f"{config.training.model}-{i}.ckpt"
            )
            print(f"{checkpoint_path=!s}")
            trainer.save_checkpoint(checkpoint_path)

    save_times = profiling_checkpointer.save_times
    model_size = get_model_size(model)
    throughput_stats = calculate_throughput(save_times, model_size)
    utilization_stats = {k: v.summarize() for k, v in monitor.resource_data.items()}
    all_stats = dict()
    all_stats["throughput"] = throughput_stats
    all_stats["utilization"] = utilization_stats
    all_stats["model_size"] = f"{model_size:.2f}MB"
    all_stats["mean_time"] = f"{save_times.summarize()['mean']}s"

    all_stats_pretty = json.dumps(all_stats, indent=2)
    print(f"{all_stats_pretty=!s}")
    return all_stats


def build_checkpoint_path(uri: str, suffix: str) -> str:
    if uri.startswith("s3://"):
        bucket, prefix = parse_s3_uri(uri)
        path = Path(bucket) / Path(prefix) / Path(suffix)
        return "s3://{0}".format(path)
    else:
        return str(Path(uri) / Path(suffix))


def get_model_size(model) -> float:
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2

    return size_all_mb


def calculate_throughput(save_times: Distribution, model_size: float) -> dict:
    latency_throughput_stat_map = {
        "min": "max",
        "max": "min",
        "p90": "p10",
        "p75": "p25",
        "mean": "mean",
    }
    return {
        latency_throughput_stat_map.get(stat, stat): "{:.3f}MB/s".format(
            model_size / val
        )
        for stat, val in save_times.summarize().items()
        if stat != "n"  # exclude the 'n' stat
    }
