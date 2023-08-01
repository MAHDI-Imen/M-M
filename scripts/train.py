from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.nn.functional import one_hot, softmax
from unet import UNet
import pytorch_lightning as pl
from einops import rearrange
from torch import argmax
import pytorch_lightning.loggers as pl_loggers
import pandas as pd

from scripts.metrics import get_metric_scores
import wandb


class LitUnet(pl.LightningModule):
    def __init__(
        self,
        model_name="Unet",
        lr=1e-3,
        num_encoding_blocks=4,
        out_channels_first_layer=32,
    ):
        super(LitUnet, self).__init__()
        self.Unet = UNet(
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

        self.lr = lr
        self.model_name = model_name
        self.criterion = CrossEntropyLoss(reduction="mean")

        self.save_hyperparameters()

    def forward(self, x):
        output = self.Unet(x)
        return output

    def training_step(self, batch, batch_idx):
        loss = self._commun_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._commun_step(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def on_test_start(self):
        self.results = pd.DataFrame(columns=["Centre", "subject_idx"])

        self.results = self.results.assign(
            Dice_BG_ED=None,
            Dice_LV_ED=None,
            Dice_MYO_ED=None,
            Dice_RV_ED=None,
            Dice_BG_ES=None,
            Dice_LV_ES=None,
            Dice_MYO_ES=None,
            Dice_RV_ES=None,
        )

        self.results = self.results.assign(
            IoU_BG_ED=None,
            IoU_LV_ED=None,
            IoU_MYO_ED=None,
            IoU_RV_ED=None,
            IoU_BG_ES=None,
            IoU_LV_ES=None,
            IoU_MYO_ES=None,
            IoU_RV_ES=None,
        )

        self.wandb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, pl_loggers.WandbLogger):
                self.wandb_logger = logger.experiment
                break

        self.table = wandb.Table(columns=["Centre", "SubjectID", "Slice", "Image"])

    def test_step(self, batch, batch_idx, dataloader_idx):
        centre = dataloader_idx
        x, y = batch
        y_pred = self._get_predictions(x)

        scores = get_metric_scores(y, y_pred)
        self.results.loc[len(self.results)] = [centre, batch_idx] + scores

        if batch_idx == 1:
            self._log_to_wandb(x, y, y_pred, centre, batch_idx)

        loss = self._commun_step(batch)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def on_test_end(self):
        self.wandb_logger.log({f"Results": self.table})
        results = self._save_results()
        return results

    def _commun_step(self, batch):
        x, y = batch
        output = self(x)
        y_one_hot = one_hot(y.long(), num_classes=4)
        y_one_hot = rearrange(y_one_hot, "b h w c -> b c h w").double()
        loss = self.criterion(output, y_one_hot)
        return loss

    def _save_results(self):
        metadata = self.trainer.datamodule.metadata.sort_values(
            by=["Centre", "SubjectID"]
        )
        self.results = self.results.set_index(metadata.index)

        self.results["Centre"] = self.results.iloc[:, 0].astype(int)

        self.results = self.results.drop(columns=["subject_idx"])

        self.results.to_csv(f"models/{self.model_name}/results.csv", index=True)
        return self.results

    def _get_predictions(self, x):
        output = self(x)
        prob = softmax(output, dim=1)
        y_pred = argmax(prob, dim=1)
        return y_pred

    def _log_to_wandb(self, images, labels, predictions, centre, batch_idx):
        class_labels = {1: "LV", 2: "MYO", 3: "RV"}

        for slice, (img, label, pred) in enumerate(zip(images, labels, predictions)):
            mask_img = wandb.Image(
                img.cpu(),
                masks={
                    "prediction": {
                        "mask_data": pred.cpu(),
                        "class_labels": class_labels,
                    },
                    "groung truth": {
                        "mask_data": label.cpu(),
                        "class_labels": class_labels,
                    },
                },
            )

            self.table.add_data(centre, batch_idx, slice, mask_img)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)


def main():
    return 0


if __name__ == "__main__":
    main()
