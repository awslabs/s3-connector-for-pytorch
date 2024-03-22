#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
import json
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from json import JSONEncoder
from typing import Dict, Any

import numpy as np
import psutil
import torch.cuda
from pynvml import (
    nvmlInit,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
)

monitor_gpu = False
if torch.cuda.is_available():
    monitor_gpu = True
    nvmlInit()


class Distribution:
    def __init__(self, initial_capacity: int, precision: int = 4):
        self.initial_capacity = initial_capacity
        self._values = np.zeros(shape=initial_capacity, dtype=np.float32)
        self._idx = 0
        self.precision = precision

    def _expand_if_needed(self):
        if self._idx > self._values.size - 1:
            self._values = np.concatenate(
                self._values, np.zeros(self.initial_capacity, dtype=np.float32)
            )

    def add(self, val: float):
        self._expand_if_needed()
        self._values[self._idx] = val
        self._idx += 1

    def summarize(self) -> dict:
        window = self._values[: self._idx]
        if window.size == 0:
            return
        return {
            "n": window.size,
            "mean": round(float(window.mean()), self.precision),
            "min": round(np.percentile(window, 0), self.precision),
            "p50": round(np.percentile(window, 50), self.precision),
            "p75": round(np.percentile(window, 75), self.precision),
            "p90": round(np.percentile(window, 90), self.precision),
            "max": round(np.percentile(window, 100), self.precision),
        }

    def __repr__(self):
        summary_str = json.dumps(self.summarize())
        return "Distribution({0})".format(summary_str)


@dataclass(frozen=True)
class ExperimentResult:
    elapsed_time: float
    volume: float
    checkpoint_times: Distribution = None
    utilization: Dict[str, Distribution] = None

    @cached_property
    def throughput(self):
        return self.volume / self.elapsed_time


class ExperimentResultJsonEncoder(JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, ExperimentResult):
            o: ExperimentResult = o
            return {
                "volume": o.volume,
                "elapsed_time": o.elapsed_time,
                "throughput": o.throughput,
                "utilization": {k: v.summarize() for k, v in o.utilization.items()},
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
        self._utilization = defaultdict(lambda: Distribution(chunk_size))
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
