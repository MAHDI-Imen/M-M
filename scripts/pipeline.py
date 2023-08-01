import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from train import LitUnet
from data import CentreDataModule

from result_analysis import save_results
import importlib
import os

import wandb

import argparse


def pipeline(config_name="config.config"):
    config = importlib.import_module(config_name)

    if not os.path.exists(f"models/{config.MODEL_NAME}"):
        os.mkdir(f"models/{config.MODEL_NAME}")

    seed_everything(config.SEED)

    wandb_logger = WandbLogger(
        project=config.PROJECT_NAME, entity=config.ENTITY, name=config.MODEL_NAME
    )

    model = LitUnet(model_name=config.MODEL_NAME, lr=config.LEARNING_RATE)

    dm = CentreDataModule(
        config.TRAINING_VENDOR,
        split_ratio=config.SPLIT_RATIO,
        load_transform=config.TRANSFORM,
        batch_size=config.BATCH_SIZE,
    )

    trainer = pl.Trainer(
        max_epochs=config.NUM_EPOCHS,
        deterministic=True,
        logger=wandb_logger,
        log_every_n_steps=1,
        enable_model_summary=False,
        callbacks=[EarlyStopping("val_loss", patience=2)],
        fast_dev_run=config.FAST_DEV_RUN,
    )

    trainer.fit(model, datamodule=dm)

    trainer.save_checkpoint(f"models/{config.MODEL_NAME}/{config.MODEL_NAME}.ckpt")

    results = trainer.test(model, ckpt_path="best", datamodule=dm)

    fig = save_results(config.MODEL_NAME)

    wandb_logger.experiment.log({f"Results/{config.MODEL_NAME}": wandb.Image(fig)})

    wandb.finish()

    print(results)


def main():
    parser = argparse.ArgumentParser(description="Run pipeline")
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        help="config file name",
        default="config",
    )

    args = parser.parse_args()

    pipeline(args.config_file)


if __name__ == "__main__":
    main()
