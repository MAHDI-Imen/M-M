import random
import os
from glob import glob
import matplotlib.pyplot as plt

from utils import load_metadata, save_metadata

from tqdm.auto import tqdm
from monai.transforms import LoadImage


def extract_pix_dim(metadata):
    """
    Extract pix_dim from the images and save it to the metadata file

    Args:
        metadata (pd.DataFrame): metadata file

    Returns:
        metadata (pd.DataFrame): metadata file with pix_dim

    """
    metadata = metadata.assign(
        time_dim=None, x_dim=None, y_dim=None, z_dim=None
    ).assign(x_pixdim=None, y_pixdim=None, z_pixdim=None)

    subject_ids = metadata.index
    subject_images = metadata.Image_path

    for subject_id, path in tqdm(
        zip(subject_ids, subject_images),
        desc="Extract pix_dim",
        total=len(subject_ids),
        unit="Subject",
    ):
        data = LoadImage(image_only=True, ensure_channel_first=True)(path)

        metadata.loc[subject_id, ["time_dim", "x_dim", "y_dim", "z_dim"]] = [
            dim for dim in data.shape
        ]
        metadata.loc[subject_id, ["x_pixdim", "y_pixdim", "z_pixdim"]] = [
            pixdim.item() for pixdim in data.pixdim
        ]

    save_metadata(metadata)

    return metadata


def save_subjects_files_paths(root_directory, metadata):
    """
    Extract the paths of the images and segmentations and save them in the metadata file.

    Args:
        root_directory (str): path to the root directory of the dataset
        metadata (pd.DataFrame): metadata file

    Returns:
        metadata (pd.DataFrame): metadata file with paths

    """
    metadata = metadata.assign(Image_path=None, Seg_path=None)

    subject_ids = list(metadata.index)

    for subject_id in tqdm(subject_ids, desc="Extract paths", unit="Subject"):
        pattern = os.path.join(root_directory, "**", subject_id, "*.nii.gz")
        files_found = glob(pattern, recursive=True)

        if not files_found:
            continue

        files_found.sort()

        image_file, label_file = files_found

        metadata.loc[subject_id, ["Image_path", "Seg_path"]] = [image_file, label_file]

    metadata = metadata[~metadata.Image_path.isna()]

    save_metadata(metadata)

    return metadata


def split_training_data(
    metadata, train_ratio=0.8, vendor="A", train_centre=0, save=False, seed=42
):
    """
    Split the training data of one vendor into train and validation sets.

    Args:
        metadata (pd.DataFrame): metadata file
        train_ratio (float): ratio of training data
        vendor (str): vendor name
        train_centre (int): new centre number for the training data
        save (bool): whether or not to save the metadata file
        seed (int): random seed

    Returns:
        metadata (pd.DataFrame): updated metadata file
    """
    random.seed(seed)
    n_total = metadata.Vendor.value_counts()[vendor]
    n_train = int(n_total * train_ratio)
    indices = metadata.index[metadata["Vendor"] == vendor].tolist()
    train_indices = random.sample(indices, n_train)

    metadata.loc[train_indices, "Centre"] = train_centre

    print(
        f"total number of samples: {n_total}, train samples: {n_train}, Validation: {n_total-n_train}"
    )
    if save:
        save_metadata(metadata)

    return metadata


def pre_process_metadata(root_directory=None):
    if root_directory is None:
        root_directory = "Data_original/OpenDataset"

    metadata = (
        load_metadata(
            os.path.join(
                root_directory,
                "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv",
            )
        )
        .rename(columns={"External code": "SubjectID"})
        .set_index("SubjectID")
    )

    metadata = save_subjects_files_paths(root_directory, metadata)

    metadata = extract_pix_dim(metadata)

    metadata = split_training_data(
        metadata, train_ratio=0.8, vendor="A", save=False, seed=42
    )

    return metadata


def plot_vendor_stats(vendor, column, kind):
    """
    Plot the distribution of a column for a given vendor.

    Args:

        vendor (str): vendor name (A, B, C, D)
        column (str): column name
        kind (str): type of plot (hist, pie)

    Example:
        >>> plot_vendor_stats("A", "x_pixdim", "hist")
    """
    metadata = load_metadata()

    data = metadata.loc[metadata.Vendor == vendor, [column]].dropna()

    if kind == "pie":
        data = data.astype("uint16")
        data = data.value_counts()
    ax = data.plot(kind=kind, title=f"{column} for vendor {vendor}")
    ax.grid(True)
    ax.set_xlabel(column)
    plt.show()


def main():
    pre_process_metadata()


if __name__ == "__main__":
    main()
