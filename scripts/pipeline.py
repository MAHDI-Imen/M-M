import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything

try:
    from train import LitUnet
except ModuleNotFoundError:
    from scripts.train import LitUnet

try:
    from data import CentreDataModule
except ModuleNotFoundError:
    from scripts.data import CentreDataModule

try:
    from result_analysis import save_results
except ModuleNotFoundError:
    from scripts.result_analysis import save_results

import importlib
import os

import wandb

import argparse


def pipeline(config_name="config.config"):
    CONFIG = importlib.import_module(config_name)

    MODEL_DIR = f"models/{CONFIG.MODEL_NAME}"
    MODEL_DIR_EXISTS = os.path.exists(MODEL_DIR)
    if not MODEL_DIR_EXISTS:
        os.mkdir(MODEL_DIR)

    seed_everything(CONFIG.SEED)

    wandb_logger = WandbLogger(
        project=CONFIG.PROJECT_NAME, entity=CONFIG.ENTITY, name=CONFIG.MODEL_NAME
    )

    model = LitUnet(model_name=CONFIG.MODEL_NAME, lr=CONFIG.LEARNING_RATE)

    dm = CentreDataModule(
        CONFIG.TRAINING_VENDOR,
        split_ratio=CONFIG.SPLIT_RATIO,
        transform=CONFIG.TRANSFORM,
        load_transform=CONFIG.LOAD_TRANSFORM,
        batch_size=CONFIG.BATCH_SIZE,
    )

    trainer = pl.Trainer(
        max_epochs=CONFIG.NUM_EPOCHS,
        deterministic=True,
        logger=wandb_logger,
        log_every_n_steps=1,
        enable_model_summary=False,
        callbacks=[EarlyStopping("val_loss", patience=CONFIG.PATIENCE)],
        fast_dev_run=CONFIG.FAST_DEV_RUN,
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=dm)

    trainer.save_checkpoint(f"{MODEL_DIR}/{CONFIG.MODEL_NAME}.ckpt")

    mean_dice_results = trainer.test(model, ckpt_path="best", datamodule=dm)

    metric_bbox_fig = save_results(CONFIG.MODEL_NAME)


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
