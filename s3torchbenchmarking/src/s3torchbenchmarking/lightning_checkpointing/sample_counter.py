from typing import Any

from lightning import Callback
import lightning.pytorch as pl


class SampleCounter(Callback):
    def __init__(self) -> None:
        super().__init__()
        self._count = 0

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        super().on_train_batch_start(trainer, pl_module, batch, batch_idx)
        self._count += len(batch)

    @property
    def count(self):
        return self._count
