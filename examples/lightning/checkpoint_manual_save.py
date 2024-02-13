#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from lightning import Trainer
from lightning.pytorch.demos import WikiText2, LightningTransformer
from torch.utils.data import DataLoader

from s3torchconnector.lightning import S3LightningCheckpoint


def main(region: str, checkpoint_path: str):
    dataset = WikiText2()
    dataloader = DataLoader(dataset, num_workers=3)

    model = LightningTransformer(vocab_size=dataset.vocab_size)
    s3_lightning_checkpoint = S3LightningCheckpoint(region)

    # No automatic checkpointing set up here.
    trainer = Trainer(
        plugins=[s3_lightning_checkpoint],
        enable_checkpointing=False,
        min_epochs=4,
        max_epochs=5,
        max_steps=3,
    )
    trainer.fit(model, dataloader)
    # Manually create checkpoint to the desired location
    trainer.save_checkpoint(checkpoint_path)


if __name__ == "__main__":
    import os

    main(os.getenv("REGION"), os.getenv("CHECKPOINT_PATH"))
