from time import perf_counter
from typing import Dict, Any, Optional

from lightning.fabric.plugins import CheckpointIO
from lightning.fabric.utilities.types import _PATH

from s3torchbenchmarking.benchmark_utils import Distribution


class CheckpointProfiler(CheckpointIO):
    def __init__(self, delegate: CheckpointIO) -> None:
        super().__init__()
        self.delegate = delegate
        self.save_times = Distribution(initial_capacity=1024)

    def load_checkpoint(
        self, path: _PATH, map_location: Optional[Any] = None
    ) -> Dict[str, Any]:
        return self.delegate.load_checkpoint(path, map_location)

    def remove_checkpoint(self, path: _PATH) -> None:
        self.delegate.remove_checkpoint(path)

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: _PATH,
        storage_options: Optional[Any] = None,
    ) -> None:
        # TODO: should we profile other operations as well?
        start_time = perf_counter()
        self.delegate.save_checkpoint(checkpoint, path, storage_options)
        elapsed_time = perf_counter() - start_time
        self.save_times.add(elapsed_time)
