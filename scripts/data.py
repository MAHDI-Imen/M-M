from torch.utils.data import Dataset, DataLoader
from monai.transforms import LoadImage
from einops import rearrange, pack
from tqdm.auto import tqdm
from pytorch_lightning import LightningDataModule
import os
from scripts.utils import load_metadata
from scripts.pre_process_metadata import split_training_data


def load_image_as_2D_slices(image_path):
    image = LoadImage(image_only=True, ensure_channel_first=True)(image_path)
    slices = rearrange(image, "time h w slices c-> (time slices) c h w")
    return slices


def load_and_transform_images(paths, transform):
    images = []
    for path in tqdm(paths, unit="File"):
        image = load_image_as_2D_slices(path)
        if transform:
            image = transform(image)
        images.append(image)
    images, ps = pack(images, "* c h w")
    return images, ps


def load_2D_data(centre, metadata=None, transform=None, target_transform=None):
    if metadata is None:
        metadata = load_metadata()
    subject_ids = list(metadata[metadata.Centre == centre].index)

    data_dir = "Data/M&Ms/OpenDataset"

    print("Loading Images for centre: ", centre)
    images_paths = [
        os.path.join(data_dir, subject_id, f"{subject_id}_sa.nii.gz")
        for subject_id in subject_ids
    ]

    print("Loading Labels for centre: ", centre)
    labels_paths = [
        os.path.join(data_dir, subject_id, f"{subject_id}_sa_gt.nii.gz")
        for subject_id in subject_ids
    ]

    images, subject_volume_sizes = load_and_transform_images(images_paths, transform)
    labels, subject_volume_sizes = load_and_transform_images(
        labels_paths, target_transform
    )
    labels = rearrange(labels, "b 1 h w -> b h w")

    return images, labels, subject_volume_sizes


class Centre2DDataset(Dataset):
    def __init__(
        self,
        centre,
        metadata,
        transform=None,
        target_transform=None,
        load_transform=None,
    ):
        self.images, self.labels, ps = load_2D_data(
            centre=centre, metadata=metadata, transform=load_transform
        )

        self.subject_volume_sizes = [sizes[0] for sizes in ps]

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        seg = self.labels[index]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            seg = self.target_transform(seg)

        return image, seg


class CentreDataModule(LightningDataModule):
    def __init__(
        self,
        train_vendor=None,
        train_centre=6,
        val_centre=3,
        split_ratio=0.7,
        transform=None,
        target_transform=None,
        load_transform=None,
        batch_size=8,
    ):
        super().__init__()
        self.train_vendor = train_vendor
        self.split_ratio = split_ratio

        self.train_centre = train_centre
        self.val_centre = val_centre

        self.transform = transform
        self.target_transform = target_transform
        self.load_transform = load_transform

        self.metadata = load_metadata()
        self.centres = list(self.metadata.Centre.unique())

        self.batch_size = batch_size

        self.test_datasets = None

    # def prepare_data(self):
    #   pre_process_metadata()
    #   extract_ROI()

    def setup(self, stage: str):
        if stage == "fit":
            if self.train_vendor:
                self.train_centre = 0

                self.val_centre = self.metadata.loc[
                    self.metadata.Vendor == self.train_vendor, "Centre"
                ][0]

                self.centres.append(self.train_centre)

                self.metadata = split_training_data(
                    self.metadata,
                    vendor=self.train_vendor,
                    train_centre=self.train_centre,
                    train_ratio=self.split_ratio,
                    seed=2,
                )
            self.train_dataset = Centre2DDataset(
                self.train_centre,
                self.metadata,
                self.transform,
                self.target_transform,
                load_transform=self.load_transform,
            )

            self.val_dataset = Centre2DDataset(
                self.val_centre,
                self.metadata,
                load_transform=self.load_transform,
            )

        if stage == "test":
            self.centres.sort()

            if self.test_datasets is None:
                self.test_datasets = [
                    Centre2DDataset(
                        centre, self.metadata, load_transform=self.load_transform
                    )
                    for centre in self.centres
                ]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )

    def test_dataloader(self):
        return [
            DataLoader(
                ds,
                batch_sampler=self.batch_sampler(ds.subject_volume_sizes),
                num_workers=4,
            )
            for ds in self.test_datasets
        ]

    def batch_sampler(self, subject_volume_sizes):
        batch_sampler = [
            [sum(subject_volume_sizes[:i]) + j for j in list(range(size))]
            for i, size in enumerate(subject_volume_sizes)
        ]
        return batch_sampler


def main():
    return 0


if __name__ == "__main__":
    main()
