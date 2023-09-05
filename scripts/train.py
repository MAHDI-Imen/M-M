import wandb
import pandas as pd
from einops import rearrange

from unet import UNet

from torch import argmax
import torch.optim as optim
from torch.nn.functional import one_hot, softmax

import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers


from monai.losses.dice import DiceLoss
from torch.nn import CrossEntropyLoss

try:
    from metrics import get_metric_scores
except ModuleNotFoundError:
    from scripts.metrics import get_metric_scores


class LitUnet(pl.LightningModule):
    def __init__(
        self,
        model_name="Unet",
        lr=1e-3,
        num_encoding_blocks=4,
        out_channels_first_layer=32,
    ):
        super(LitUnet, self).__init__()

        self.Unet = self._instantiate_Unet(
            num_encoding_blocks, out_channels_first_layer
        )

        self.criterion = DiceLoss(
            reduction="mean", squared_pred=True, include_background=True, softmax=True
        )

        self.lr = lr
        self.model_name = model_name
        self.wandb_logger = None

    def forward(self, x):
        output = self.Unet(x)
        return output

    def on_train_start(self):
        self._setup_wandb_logger()
        self.table = wandb.Table(columns=["BatchID", "Image"])

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self._get_loss(x, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        if batch_idx % 30 == 0:
            y_pred = self._get_predicted_segmentations(x)
            self._add_row_to_wandb_table(x, y, y_pred, batch_idx)
        return loss

    def on_train_end(self):
        self.wandb_logger.log({f"Train/{self.model_name}": self.table})

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self._get_loss(x, y)
        y_pred = self._get_predicted_segmentations(x)

        scores = get_metric_scores(y, y_pred)

        log_dict = self._get_metrics_to_log(scores, loss)
        self.log_dict(log_dict, prog_bar=True, on_epoch=True, on_step=False)

        return log_dict

    def on_test_start(self):
        if self.wandb_logger is None:
            self._setup_wandb_logger()

        self._setup_results_df()
        self.table = wandb.Table(columns=["Centre", "Image"])

    def test_step(self, batch, batch_idx, dataloader_idx):
        centre = dataloader_idx
        x, y = batch
        y_pred = self._get_predicted_segmentations(x)

        scores = get_metric_scores(y, y_pred)
        self.results.loc[len(self.results)] = [centre, batch_idx] + scores

        if batch_idx == 1:
            self._add_row_to_wandb_table(x, y, y_pred, centre)

        mean_dice = self._get_metrics_to_log(scores, mean_value=True)
        self.log("mean_dice", mean_dice, prog_bar=True, on_epoch=True, on_step=False)

        return mean_dice

    def on_test_end(self):
        self.wandb_logger.log({self.model_name: self.table})
        results = self._save_results()

        self.save_hyperparameters()

        return results

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, verbose=True, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def _save_results(self):
        metadata = self.trainer.datamodule.metadata.sort_values(
            by=["Centre", "SubjectID"]
        )
        self.results = self.results.set_index(metadata.index)
        self.results["Centre"] = self.results.iloc[:, 0].astype(int)
        self.results = self.results.drop(columns=["subject_idx"])
        self.results.to_csv(f"models/{self.model_name}/results.csv", index=True)
        return self.results

    def _get_loss(self, x, y):
        output = self(x)
        y_one_hot = one_hot(y.long(), num_classes=4)
        y_one_hot = rearrange(y_one_hot, "b h w c -> b c h w").double()
        loss = self.criterion(y_one_hot, output)
        return loss

    def _get_predicted_segmentations(self, x):
        output = self(x)
        prob = softmax(output, dim=1)
        y_pred = argmax(prob, dim=1)
        return y_pred

    def _add_row_to_wandb_table(self, images, labels, predictions, index):
        class_labels = {1: "LV", 2: "MYO", 3: "RV"}

        for img, label, pred in zip(images, labels, predictions):
            masks = {
                "groung truth": {
                    "mask_data": label.cpu(),
                    "class_labels": class_labels,
                },
                "prediction": {
                    "mask_data": pred.cpu(),
                    "class_labels": class_labels,
                },
            }

            mask_img = wandb.Image(
                img.cpu(),
                masks=masks,
            )

            self.table.add_data(index, mask_img)

    def _instantiate_Unet(self, num_encoding_blocks, out_channels_first_layer):
        unet_model = UNet(
            in_channels=1,
            out_classes=4,
            dimensions=2,
            num_encoding_blocks=num_encoding_blocks,
            out_channels_first_layer=out_channels_first_layer,
            normalization="batch",
            upsampling_type="conv",
            padding=True,
            activation="PReLU",
        )
        return unet_model

    def _setup_wandb_logger(self):
        for logger in self.trainer.loggers:
            if isinstance(logger, pl_loggers.WandbLogger):
                self.wandb_logger = logger.experiment
                break

    def _get_metrics_to_log(self, scores, loss=None, mean_value=False):
        ed_lv_dice, ed_myo_dice, ed_rv_dice = scores[1:4]
        es_lv_dice, es_myo_dice, es_rv_dice = scores[5:8]

        log_dict = {
            "ed_lv_dice": ed_lv_dice,
            "es_lv_dice": es_lv_dice,
            "ed_myo_dice": ed_myo_dice,
            "es_myo_dice": es_myo_dice,
            "ed_rv_dice": ed_rv_dice,
            "es_rv_dice": es_rv_dice,
        }

        if loss:
            log_dict["val_loss"] = loss

        if mean_value:
            mean_dice = sum(log_dict.values()) / len(log_dict)
            return mean_dice

        return log_dict

    def _setup_results_df(self):
        self.results = pd.DataFrame(
            columns=[
                "Centre",
                "subject_idx",
                "Dice_BG_ED",
                "Dice_LV_ED",
                "Dice_MYO_ED",
                "Dice_RV_ED",
                "Dice_BG_ES",
                "Dice_LV_ES",
                "Dice_MYO_ES",
                "Dice_RV_ES",
                "IoU_BG_ED",
                "IoU_LV_ED",
                "IoU_MYO_ED",
                "IoU_RV_ED",
                "IoU_BG_ES",
                "IoU_LV_ES",
                "IoU_MYO_ES",
                "IoU_RV_ES",
            ]
        )


def main():
    return 0


if __name__ == "__main__":
    main()
