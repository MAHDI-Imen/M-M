from torch.utils.data import Dataset
import os
import torchio as tio
from monai.transforms import LoadImage
from einops import rearrange, pack
from tqdm import tqdm

from scripts.utils import get_subjects_files_paths, load_metadata


def load_as_2D_slices(image_path):
    image = LoadImage(image_only=True, ensure_channel_first=True)(image_path)
    slices = rearrange(image, "time h w slices c-> (time slices) c h w")
    return slices


def get_channel_index(cardiac_phase, subject_id, metadata=None):
    if metadata is None:
        metadata = load_metadata()
    index = metadata.loc[metadata["External code"] == subject_id, cardiac_phase].iloc[0]
    return index


def load_centre_2D_data(centre, root_directory, metadata=None, transform=None):
    subject_ids, images_paths, labels_paths = get_subjects_files_paths(
        root_directory, centre=centre, metadata=metadata
    )

    if metadata is None:
        metadata = load_metadata()

    images = []
    labels = []
    print("Loading Files")
    for subject_id, image_path, labels_path in tqdm(
        zip(subject_ids, images_paths, labels_paths)
    ):
        try:
            image = load_as_2D_slices(image_path)
            seg = load_as_2D_slices(labels_path)

            if transform:
                image = transform(image)
                seg = transform(seg)

            images.append(image)
            labels.append(seg)
        except:
            print("Couldn't load:", subject_id)
            continue

    return images, labels


def get_centre_2D_dataset(centre, root_directory, transform=None, metadata=None):
    if metadata is None:
        metadata = load_metadata()

    images, labels = load_centre_2D_data(centre, root_directory, metadata, transform)
    images, ps = pack(images, "* c h w")
    labels, ps = pack(labels, "* c h w")
    labels = rearrange(labels, "b 1 h w -> b h w")
    return images, labels, ps


class Centre2DDataset(Dataset):
    def __init__(self, centre_data, transform=None):
        self.images, self.labels = centre_data
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        seg = self.labels[index]

        if self.transform:
            image = self.transform(image)
            seg = self.transform(seg)

        return image, seg


def main():
    return 0


if __name__ == "__main__":
    main()
