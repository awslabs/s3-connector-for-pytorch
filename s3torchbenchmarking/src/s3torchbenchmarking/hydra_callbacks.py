import json
from pathlib import Path
from typing import Any, List, Optional

from hydra.core.utils import JobReturn
from hydra.experimental.callback import Callback
from omegaconf import DictConfig, OmegaConf


class ResultCollatingCallback(Callback):
    def __init__(self):
        self.multirun_dir = Optional[Path]
        self.job_returns: List[JobReturn] = []

    def on_job_end(
        self, config: DictConfig, job_return: JobReturn, **kwargs: Any
    ) -> None:
        """
        Runtime variables like the output directory are not available when `on_multirun_end` is called, but they
        are available when this method is called. So we collect them here and refer to this common state later.
        """
        self.job_returns.append(job_return)
        self.multirun_dir = Path(
            job_return.hydra_cfg["hydra"]["runtime"]["output_dir"]
        ).parent

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        collated_results = self._collate_results()
        results_path = self._write_results(collated_results)
        print(f"Collated results written to: {results_path}")

    def _collate_results(self) -> List:
        collated_results = []
        for job_return in self.job_returns:
            job_output_dir = Path(
                job_return.hydra_cfg["hydra"]["runtime"]["output_dir"]
            )
            job_result_path = job_output_dir / "result.json"
            with open(job_result_path) as infile:
                item = {
                    "job_id": job_return.hydra_cfg["hydra"]["job"]["id"],
                    "cfg": OmegaConf.to_container(job_return.cfg),
                    "result": json.load(infile),
                }
                collated_results.append(item)
        return collated_results

    def _write_results(self, collated_results: List) -> Path:
        results_path = self.multirun_dir / "collated_results.json"
        with open(results_path, "w") as outfile:
            json.dump(collated_results, outfile)

        return results_path
