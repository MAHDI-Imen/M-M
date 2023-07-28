from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.nn.functional import one_hot, softmax
from unet import UNet
import pytorch_lightning as pl
from einops import rearrange
from torch import argmax
import pytorch_lightning.loggers as pl_loggers
import matplotlib.pyplot as plt


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

    def training_step(self, batch, batch_idx):
        loss = self._commun_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._commun_step(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        output = self(x)
        prob = softmax(output, dim=1)
        y_pred = argmax(prob, dim=1)
        loss = self._commun_step(batch)
        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )
        if batch_idx % 10:
            self.log_tb_images(x, y, y_pred, batch_idx, dataloader_idx)

        return loss

    def _commun_step(self, batch):
        x, y = batch

        output = self(x)

        y_one_hot = one_hot(y.long(), num_classes=4)
        y_one_hot = rearrange(y_one_hot, "b h w c -> b c h w").double()

        loss = self.criterion(output, y_one_hot)
        return loss

    def log_tb_images(self, images, labels, predictions, batch_idx, dataloader_idx):
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, pl_loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
            raise ValueError("TensorBoard Logger not found")
        fig = create_figure(images, labels, predictions)
        tb_logger.add_figure(f"Centre_{dataloader_idx}_Results", fig, batch_idx)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)


def display_subplot(data, title, ax, cmap="gray"):
    ax.imshow(data, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")


def create_figure(images, labels, predictions):
    fig, axes = plt.subplots(3, 1, figsize=(len(images), 5))
    fig.suptitle(f"Image Predictions and Ground Truth", fontsize=16)

    display_subplot(rearrange(images.cpu(), "b c h w -> h (b w) c"), "Images", axes[0])
    display_subplot(
        rearrange(labels.cpu(), "b h w -> h (b w)"),
        "Ground Truth",
        axes[1],
        cmap="viridis",
    )
    display_subplot(
        rearrange(predictions.cpu(), "b h w -> h (b w)"),
        "Predictions",
        axes[2],
        cmap="viridis",
    )
    return fig


def main():
    return 0


if __name__ == "__main__":
    main()
