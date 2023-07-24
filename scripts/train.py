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

        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, x):
        output = self.Unet(x)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        probabilities = F.softmax(output, dim=1)
        y_one_hot = F.one_hot(y.long(), num_classes=4)
        y_one_hot = rearrange(y_one_hot, "b h w c -> b c h w").float()
        loss = self.criterion(probabilities, y_one_hot)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        output = self(x)
        probabilities = F.softmax(output, dim=1)

        y_one_hot = F.one_hot(y.long(), num_classes=4)
        y_one_hot = rearrange(y_one_hot, "b h w c -> b c h w").float()

        loss = self.criterion(probabilities, y_one_hot)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)


def main():
    return 0


if __name__ == "__main__":
    main()
