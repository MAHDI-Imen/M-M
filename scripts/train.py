import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from unet import UNet
import pytorch_lightning as pl
from einops import rearrange


class LitUnet(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super(LitUnet, self).__init__()
        self.Unet = UNet(
            in_channels=1,
            out_classes=4,
            dimensions=2,
            num_encoding_blocks=3,
            out_channels_first_layer=32,
            normalization="batch",
            upsampling_type="conv",
            padding=True,
            activation="PReLU",
        )

        self.lr = lr

        self.criterion = nn.CrossEntropyLoss(reduction="mean")

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

    def _commun_step(self, batch):
        x, y = batch

        output = self(x)
        probabilities = F.softmax(output, dim=1)

        y_one_hot = F.one_hot(y.long(), num_classes=4)
        y_one_hot = rearrange(y_one_hot, "b h w c -> b c h w").double()

        loss = self.criterion(probabilities, y_one_hot)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)


def main():
    return 0


if __name__ == "__main__":
    main()
