from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.nn.functional import one_hot, softmax
from unet import UNet
import pytorch_lightning as pl
from einops import rearrange
from torch import argmax
import pytorch_lightning.loggers as pl_loggers
import matplotlib.pyplot as plt
import pandas as pd

from miseval import evaluate


class LitUnet(pl.LightningModule):
    def __init__(
        self,
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

        self.criterion = CrossEntropyLoss(reduction="mean")

        self.save_hyperparameters()

    def forward(self, x):
        output = self.Unet(x)
        return output

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

    def on_test_end(self):
        self.results.iloc[:, :2] = self.results.iloc[:, :2].astype(int)
        self.results.to_csv("results.csv", index=False)

    def training_step(self, batch, batch_idx):
        loss = self._commun_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._commun_step(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx):
        centre = dataloader_idx
        x, y = batch
        y_pred = self.get_predictions(x)

        n_slices = y.shape[0] // 2

        ed_labels, ed_predictions = y[:n_slices], y_pred[:n_slices]
        es_labels, es_predictions = y[n_slices:], y_pred[n_slices:]

        ED_scores = get_scores(ed_labels, ed_predictions)
        ES_scores = get_scores(es_labels, es_predictions)

        self.results.loc[len(self.results)] = (
            [centre, batch_idx] + ED_scores + ES_scores
        )
        if batch_idx % 10:
            self.log_tb_example(x, y, y_pred, batch_idx, centre)

        loss = self._commun_step(batch)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def _commun_step(self, batch):
        x, y = batch
        output = self(x)
        y_one_hot = one_hot(y.long(), num_classes=4)
        y_one_hot = rearrange(y_one_hot, "b h w c -> b c h w").double()
        loss = self.criterion(output, y_one_hot)
        return loss

    def get_predictions(self, x):
        output = self(x)
        prob = softmax(output, dim=1)
        y_pred = argmax(prob, dim=1)
        return y_pred

    def log_tb_example(self, images, labels, predictions, batch_idx, centre):
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, pl_loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
            raise ValueError("TensorBoard Logger not found")
        fig = create_figure(images, labels, predictions)
        tb_logger.add_figure(f"Centre_{centre}_Results", fig, batch_idx)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)


def get_scores(y, y_pred):
    dice_scores = evaluate(y, y_pred, metric="Dice", multiclass=True, n_classes=4)
    IoU_scores = evaluate(y, y_pred, metric="IoU", multiclass=True, n_classes=4)
    return list(dice_scores) + list(IoU_scores)


def display_subplot(data, title, ax, cmap="gray"):
    ax.imshow(data, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")


def create_figure(images, labels, predictions):
    fig, axes = plt.subplots(3, 1, figsize=(len(images), 5))
    fig.suptitle(f"Image Predictions and Ground Truth", fontsize=16)

    display_subplot(
        rearrange(images.cpu(), "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=2),
        "Images",
        axes[0],
    )
    display_subplot(
        rearrange(labels.cpu(), "(b1 b2) h w -> (b1 h) (b2 w)", b1=2),
        "Ground Truth",
        axes[1],
        cmap="viridis",
    )
    display_subplot(
        rearrange(predictions.cpu(), "(b1 b2) h w -> (b1 h) (b2 w)", b1=2),
        "Predictions",
        axes[2],
        cmap="viridis",
    )
    return fig


def main():
    return 0


if __name__ == "__main__":
    main()
