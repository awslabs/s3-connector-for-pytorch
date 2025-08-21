#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
import random
import string
import threading
import time
from collections import defaultdict
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, TypedDict

import numpy as np
import psutil
import torch.cuda
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


class ExperimentResult(TypedDict, total=False):
    training_duration_s: float
    epoch_durations_s: List[float]
    volume: int
    checkpoint_times: Optional[List[float]]
    utilization: Dict[str, Distribution]


class ResourceMonitor:
    """
    Monitors CPU, GPU usage and memory.
    Set sleep_time_s carefully to avoid perf degradations.
    """

    def __init__(self, sleep_time_s: float = 0.05, chunk_size: int = 25_000):
        self.monitor_thread = None
        self._utilization: Dict[str, Distribution] = defaultdict(
            lambda: Distribution(chunk_size)
        )
        self.stop_event = threading.Event()
        self.sleep_time_s = sleep_time_s
        self.num_of_gpus = torch.cuda.device_count() if monitor_gpu else 0
        self.chunk_size = chunk_size

    def _monitor(self):
        while not self.stop_event.is_set():
            self._utilization["cpu_util"].add(psutil.cpu_percent())
            self._utilization["cpu_mem"].add(psutil.virtual_memory().percent)

            if monitor_gpu:
                for gpu_id in range(self.num_of_gpus):
                    gpu_info = nvmlDeviceGetUtilizationRates(
                        nvmlDeviceGetHandleByIndex(gpu_id)
                    )
                    gpu_mem_info = nvmlDeviceGetMemoryInfo(
                        nvmlDeviceGetHandleByIndex(gpu_id)
                    )
                    self._utilization[f"gpu_{gpu_id}_util"].add(gpu_info.gpu)
                    self._utilization[f"gpu_{gpu_id}_util"].add(
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


def build_random_suffix() -> str:
    """Generates a unique suffix combining timestamp with random characters for use in filepaths or S3 URIs."""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    random_suffix = "".join(random.choices(string.ascii_letters, k=4))
    return f"{timestamp}-{random_suffix}"


def build_checkpoint_path(path: str, suffix: str) -> str:
    return str(Path(path) / suffix)


def build_checkpoint_uri(uri: str, suffix: str) -> str:
    return uri.removesuffix("/") + "/" + suffix.removeprefix("/")
