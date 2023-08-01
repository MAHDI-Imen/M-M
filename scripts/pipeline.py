import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from scripts.train import LitUnet
from scripts.data import CentreDataModule
from monai.transforms import ScaleIntensity

from scripts.config import *
from scripts.result_analysis import save_results
import importlib

import wandb


def pipeline(config_name="scripts.config"):
    config = importlib.import_module(config_name)

    seed_everything(config.SEED)

    wandb_logger = WandbLogger(
        project=config.PROJECT_NAME, entity=config.ENTITY, name=config.MODEL_NAME
    )

    model = LitUnet(model_name=config.MODEL_NAME, lr=config.LEARNING_RATE)

    transform = ScaleIntensity(minv=0.0, maxv=1.0, channel_wise=True)

    dm = CentreDataModule(
        config.TRAINING_VENDOR,
        split_ratio=config.SPLIT_RATIO,
        load_transform=transform,
        batch_size=config.BATCH_SIZE,
        fast_dev_run=config.FAST_DEV_RUN,
    )

    trainer = pl.Trainer(
        max_epochs=config.NUM_EPOCHS,
        deterministic=True,
        logger=wandb_logger,
        log_every_n_steps=1,
        enable_model_summary=False,
        callbacks=[EarlyStopping("val_loss", patience=2)],
    )

    trainer.fit(model, datamodule=dm)

    trainer.save_checkpoint(f"models/{config.MODEL_NAME}/{config.MODEL_NAME}.ckpt")

    trainer.test(model, ckpt_path="best", datamodule=dm)

    results = save_results(config.MODEL_NAME)

    wandb.finish()

    print(results)


def main():
    pipeline()


if __name__ == "__main__":
    main()
