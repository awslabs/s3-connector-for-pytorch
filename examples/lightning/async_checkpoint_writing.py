#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos import WikiText2, LightningTransformer
from lightning.pytorch.plugins import AsyncCheckpointIO
from torch.utils.data import DataLoader

from s3torchconnector.lightning import S3LightningCheckpoint


def main(region: str, checkpoint_path: str):
    dataset = WikiText2()
    dataloader = DataLoader(dataset, num_workers=3)

    model = LightningTransformer(vocab_size=dataset.vocab_size)
    s3_lightning_checkpoint = S3LightningCheckpoint(region)
    async_checkpoint = AsyncCheckpointIO(s3_lightning_checkpoint)

    # This will create one checkpoint per 'step', which we define later to be 8.
    # To checkpoint more or less often, change `every_n_train_steps`.
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        save_top_k=-1,
        every_n_train_steps=1,
        filename="checkpoint-{epoch:02d}-{step:02d}",
        enable_version_counter=True,
    )

    trainer = Trainer(
        plugins=[async_checkpoint],
        callbacks=[checkpoint_callback],
        min_epochs=4,
        max_epochs=8,
        max_steps=8,
    )
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    import os

    main(os.getenv("REGION"), os.getenv("CHECKPOINT_PATH"))
