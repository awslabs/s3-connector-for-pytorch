#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
import threading
import time

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


class ExperimentResult:
    def __init__(
        self, training_time: float, num_samples: int, checkpoint_results: [] = None
    ):
        self.training_time = training_time
        self.num_samples = num_samples
        self.throughput = self.num_samples / self.training_time
        self.resource_data = {}
        self.avg_resource_data = {}
        self.checkpoint_results = checkpoint_results


class ResourceMonitor:
    """
    Monitors CPU, GPU usage and memory.
    Set sleep_time_s carefully to avoid perf degradations.
    """

    def __init__(
        self, sleep_time_s: float = 0.05, gpu_device: int = 0, chunk_size: int = 25_000
    ):
        self.monitor_thread = None
        self.resource_data = {
            "cpu_util": np.zeros(chunk_size),
            "cpu_mem": np.zeros(chunk_size),
            "gpu_util": np.zeros(chunk_size),
            "gpu_mem": np.zeros(chunk_size),
        }
        self.stop_event = threading.Event()
        self.sleep_time_s = sleep_time_s
        self.gpu_device = gpu_device
        self.chunk_size = chunk_size
        self.cur_index = 0

    def _check_and_expand(self):
        if self.cur_index >= len(self.resource_data["cpu_util"]) - 1:
            for key, arr in self.resource_data.items():
                self.resource_data[key] = np.concatenate(
                    (arr, np.zeros(self.chunk_size))
                )

    def _monitor(self):
        while not self.stop_event.is_set():
            self._check_and_expand()
            self.resource_data["cpu_util"][self.cur_index] = psutil.cpu_percent()
            self.resource_data["cpu_mem"][
                self.cur_index
            ] = psutil.virtual_memory().percent

            if monitor_gpu:
                gpu_info = nvmlDeviceGetUtilizationRates(
                    nvmlDeviceGetHandleByIndex(self.gpu_device)
                )
                gpu_mem_info = nvmlDeviceGetMemoryInfo(
                    nvmlDeviceGetHandleByIndex(self.gpu_device)
                )
                self.resource_data["gpu_util"][self.cur_index] = gpu_info.gpu
                self.resource_data["gpu_mem"][self.cur_index] = (
                    gpu_mem_info.used / gpu_mem_info.total * 100
                )
            self.cur_index += 1
            time.sleep(self.sleep_time_s)

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

        for key, arr in self.resource_data.items():
            self.resource_data[key] = arr[: self.cur_index]

    def get_full_data(self):
        return self.resource_data

    def _calculate_avg(self, start_idx: int):
        mean_measurements = {}
        for key, arr in self.resource_data.items():
            last_vals = arr[start_idx : self.cur_index]
            mean_measurements[key] = np.mean(last_vals)

        return mean_measurements

    def get_running_avg(self, window_size=50):
        start_index = max(0, self.cur_index - window_size)

        return self._calculate_avg(start_index)

    def get_avg_data(self):
        return self._calculate_avg(0)
