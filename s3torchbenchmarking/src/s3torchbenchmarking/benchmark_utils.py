#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
import json
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from json import JSONEncoder
from typing import Dict, Any, Optional
from collections import deque

import numpy as np
import psutil
import torch.cuda
from PIL import Image
from pynvml import (  # type: ignore
    nvmlInit,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
)
from torchvision.transforms import v2  # type: ignore

monitor_gpu = False
if torch.cuda.is_available():
    monitor_gpu = True
    nvmlInit()


class Distribution:
    def __init__(self, initial_capacity: int, precision: int = 4):
        self.initial_capacity = initial_capacity
        self._values = deque(maxlen=None)
        self.precision = precision

    def add(self, val: float):
        self._values.append(val)

    def summarize(self) -> dict:
        if not self._values:
            return {}
        window = np.array(self._values)
        return {
            "n": len(window),
            "mean": round(float(window.mean()), self.precision),
            "min": round(np.percentile(window, 0), self.precision),
            "p50": round(np.percentile(window, 50), self.precision),
            "p75": round(np.percentile(window, 75), self.precision),
            "p90": round(np.percentile(window, 90), self.precision),
            "max": round(np.percentile(window, 100), self.precision),
        }

    def __str__(self):
        summary_str = json.dumps(self.summarize(), indent=2)
        return "Distribution({0})".format(summary_str)


@dataclass(repr=False)
class ExperimentResult:
    elapsed_time: float
    volume: int
    checkpoint_times: Optional[Distribution] = None
    utilization: Optional[Dict[str, Distribution]] = None

    def throughput(self):
        return self.volume / self.elapsed_time

    def summarized_utilization(self):
        summary = {k: v.summarize() for k, v in self.utilization.items()}
        return json.dumps(summary, indent=2)

    def __str__(self):
        return (
            "ExperimentResult["
            "\n\ttraining_time: {0:.4f} seconds"
            "\n\tthroughput: {1:.4f} samples/second"
            "\n\tutilization:"
            "\n\t\t{2}"
            "\n\tcheckpoint_times:"
            "\n\t\t{3}"
            "\n]".format(
                self.elapsed_time,
                self.throughput(),
                self.summarized_utilization(),
                self.checkpoint_times,
            )
        )


class ExperimentResultJsonEncoder(JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, ExperimentResult):
            result: ExperimentResult = o
            utilization: Optional[Dict[str, Distribution]] = result.utilization
            if utilization is not None:
                summarized_utilization = {
                    k: v.summarize() for k, v in utilization.items()
                }
            else:
                summarized_utilization = {}
            return {
                "volume": result.volume,
                "elapsed_time": result.elapsed_time,
                "throughput": result.throughput(),
                "utilization": summarized_utilization,
            }
        return super().default(o)


class ResourceMonitor:
    """
    Monitors CPU, GPU usage and memory.
    Set sleep_time_s carefully to avoid perf degradations.
    """

    def __init__(
        self, sleep_time_s: float = 0.05, gpu_device: int = 0, chunk_size: int = 25_000
    ):
        self.monitor_thread = None
        self._utilization: Dict[str, Distribution] = defaultdict(
            lambda: Distribution(chunk_size)
        )
        self.stop_event = threading.Event()
        self.sleep_time_s = sleep_time_s
        self.gpu_device = gpu_device
        self.chunk_size = chunk_size

    def _monitor(self):
        while not self.stop_event.is_set():
            self._utilization["cpu_util"].add(psutil.cpu_percent())
            self._utilization["cpu_mem"].add(psutil.virtual_memory().percent)

            if monitor_gpu:
                gpu_info = nvmlDeviceGetUtilizationRates(
                    nvmlDeviceGetHandleByIndex(self.gpu_device)
                )
                gpu_mem_info = nvmlDeviceGetMemoryInfo(
                    nvmlDeviceGetHandleByIndex(self.gpu_device)
                )
                self._utilization["gpu_util"].add(gpu_info.gpu)
                self._utilization["gpu_mem"].add(
                    gpu_mem_info.used / gpu_mem_info.total * 100
                )
            time.sleep(self.sleep_time_s)

    @property
    def resource_data(self):
        return dict(self._utilization)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        self.monitor_thread = threading.Thread(target=self._monitor)
        self.monitor_thread.start()

    def stop(self):
        self.stop_event.set()
        self.monitor_thread.join()


class Transforms:
    IMG_TRANSFORMS = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.RandomResizedCrop((224, 224), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    @staticmethod
    def transform_image(data):
        img = Image.open(data)
        return Transforms.IMG_TRANSFORMS(img)
