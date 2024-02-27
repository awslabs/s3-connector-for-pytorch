#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos import WikiText2, LightningTransformer
from torch.utils.data import DataLoader

from s3torchconnector.lightning import S3LightningCheckpoint


def main(region: str, checkpoint_path: str, save_only_latest: bool):
    dataset = WikiText2()
    dataloader = DataLoader(dataset, num_workers=3)

    model = LightningTransformer(vocab_size=dataset.vocab_size)
    s3_lightning_checkpoint = S3LightningCheckpoint(region)

    # Save once per step, and if `save_only_latest`, replace the last checkpoint each time.
    # Replacing is implemented by saving the new checkpoint, and then deleting the previous one.
    # If `save_only_latest` is False, a new checkpoint is created for each step.
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        save_top_k=1 if save_only_latest else -1,
        every_n_train_steps=1,
        filename="checkpoint-{epoch:02d}-{step:02d}",
        enable_version_counter=True,
    )

    trainer = Trainer(
        plugins=[s3_lightning_checkpoint],
        callbacks=[checkpoint_callback],
        min_epochs=4,
        max_epochs=5,
        max_steps=3,
    )
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    import os

    main(
        os.getenv("REGION"),
        os.getenv("CHECKPOINT_PATH"),
        os.getenv("LATEST_CHECKPOINT_ONLY") == "1",
    )
