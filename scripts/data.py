from torch.utils.data import Dataset
from monai.transforms import LoadImage
from einops import rearrange, pack
from tqdm.auto import tqdm

import os

from scripts.utils import load_metadata


def load_image_as_2D_slices(image_path):
    image = LoadImage(image_only=True, ensure_channel_first=True)(image_path)
    slices = rearrange(image, "time h w slices c-> (time slices) c h w")
    return slices


def load_and_transform_images(paths, transform):
    images = []
    for path in tqdm(paths):
        image = load_image_as_2D_slices(path)
        if transform:
            image = transform(image)
        images.append(image)
    images, _ = pack(images, "* c h w")
    return images


def load_2D_data(centre, metadata=None, transform=None, target_transform=None):
    if metadata is None:
        metadata = load_metadata()
    subject_ids = list(metadata[metadata.Centre == centre].index)

    data_dir = "Data/M&Ms/OpenDataset"

    images_paths = [
        os.path.join(data_dir, subject_id, f"{subject_id}_sa.nii.gz")
        for subject_id in subject_ids
    ]

    labels_paths = [
        os.path.join(data_dir, subject_id, f"{subject_id}_sa_gt.nii.gz")
        for subject_id in subject_ids
    ]

    images = load_and_transform_images(images_paths, transform)
    labels = load_and_transform_images(labels_paths, target_transform)
    labels = rearrange(labels, "b 1 h w -> b h w")

    return images, labels


class Centre2DDataset(Dataset):
    def __init__(self, centre, metadata, transform=None, target_transform=None):
        self.images, self.labels = load_2D_data(centre=centre, metadata=metadata)

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


def main():
    return 0


if __name__ == "__main__":
    main()
