import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from unet import UNet
import pytorch_lightning as pl
from einops import rearrange
from scripts.pre_process_metadata import split_training_data
from scripts.utils import load_metadata
from scripts.data import Centre2DDataset
from torch.utils.data import DataLoader


class CentreDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_vendor=None,
        train_centre=6,
        val_centre=3,
        split_ratio=0.8,
        transform=None,
        target_transform=None,
        batch_size=8,
    ):
        super().__init__()
        self.train_vendor = train_vendor
        self.split_ratio = split_ratio

        self.train_centre = train_centre
        self.val_centre = val_centre

        self.transform = transform
        self.target_transform = target_transform

        self.metadata = load_metadata()
        self.centres = self.metadata.Centre.unique()

        self.batch_size = batch_size

    # def prepare_data(self):
    #   pre_process_metadata()
    #   extract_ROI()

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            if self.train_vendor:
                self.train_centre = max(self.centres) + 1
                self.val_centre = self.metadata.loc[
                    self.metadata.Vendor == self.train_vendor, "Centre"
                ][0]

                self.metadata = split_training_data(
                    self.metadata,
                    vendor=self.train_vendor,
                    train_centre=self.train_centre,
                    train_ratio=self.split_ratio,
                )

            self.train_dataset = Centre2DDataset(
                self.train_centre, self.metadata, self.transform, self.target_transform
            )

            self.val_dataset = Centre2DDataset(
                self.val_centre, self.metadata, self.transform, self.target_transform
            )

        # # Assign test dataset for use in dataloader(s)
        # if stage == "test":
        #     self.mnist_test = MNIST(
        #         self.data_dir, train=False, transform=self.transform
        #     )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

    # def test_dataloader(self):
    #     return DataLoader(self.mnist_test, batch_size=32)


class LitUnet(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super(LitUnet, self).__init__()
        self.Unet = UNet(
            in_channels=1,
            out_classes=4,
            dimensions=2,
            num_encoding_blocks=4,
            out_channels_first_layer=64,
            normalization="batch",
            upsampling_type="conv",
            padding=True,
            activation="PReLU",
        )

        self.lr = lr

        self.criterion = nn.CrossEntropyLoss(reduction="mean")
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

    def _commun_step(self, batch):
        x, y = batch

        output = self(x)

        y_one_hot = F.one_hot(y.long(), num_classes=4)
        y_one_hot = rearrange(y_one_hot, "b h w c -> b c h w").double()

        loss = self.criterion(output, y_one_hot)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)


def main():
    return 0


if __name__ == "__main__":
    main()
