import pandas as pd
import torch
import torchio as tio
import numpy as np
import random
import toml
from scripts.preprocessing import split_training_data
from scripts.extract_ROI import extract_ROI
from scripts.data import VendorDataset
from scripts.train import *
from scripts.load_data import load_vendor_3D
from scripts.analysis import *


def main_init():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    print("TorchIO version:", tio.__version__)
    print("Device:", device)
    return device


def perform_ROI_exctraction(padding_size, crop_size):
    data_dir = "Data_original/OpenDataset/"

    metadata_path = "Data/M&Ms/OpenDataset/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"

    folders = [
        ("Training/Labeled/", "Labeled/"),
        ("Testing/", "Testing/"),
        ("Validation/", "Validation/"),
    ]

    for source, dest in folders:
        extract_ROI(
            data_dir + source,
            data_dir + dest,
            metadata_path,
            crop_size=crop_size,
            padding_size=padding_size,
        )


def load_transforms(config, augmentation_type):
    transforms = {}
    for transform, kwargs in config[augmentation_type].items():
        probability = kwargs.pop("probability")

        exec(f"transforms[tio.transforms.{transform}(**kwargs)]=probability")
    return transforms


def load_config(config_path):
    config = toml.load(config_path)

    for key, value in config["Model"].items():
        if isinstance(value, str):
            value = f'"{value}"'
        exec(f"{key}={value}", globals())

    if config["ROI"]["extract_ROI"] == True:
        print("Start ROI extraction")
        padding_size = config["ROI"]["padding_size"]
        crop_size = config["ROI"]["crop_size"]
        perform_ROI_exctraction(padding_size, crop_size)
        print("ROI extraction performed")

    augmentation_transforms = []
    if "Spatial" in config.keys():
        spatial_transforms = load_transforms(config, "Spatial")
        augmentation_transforms.append(tio.transforms.OneOf(spatial_transforms))

    if "Intensity" in config.keys():
        intensity_transforms = load_transforms(config, "Intensity")
        augmentation_transforms.append(tio.transforms.OneOf(intensity_transforms))

    augmentation_transforms = tio.Compose(augmentation_transforms)

    return augmentation_transforms


def load_metadata(train_ratio=0.8):
    metadata_path = "Data/M&Ms/OpenDataset/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"
    metadata = pd.read_csv(metadata_path, index_col=1).drop(columns="Unnamed: 0")

    metadata = split_training_data(metadata, train_ratio=0.8)
    return metadata


def main_pipeline(config_path):
    device = main_init()

    metadata = load_metadata()

    augmentation_transforms = load_config(config_path)

    train_dataset = VendorDataset(
        "F", metadata, augmentation_transform=augmentation_transforms
    )
    valid_dataset = VendorDataset("A", metadata)

    print("Train dataset length:", len(train_dataset))
    print("Validation dataset length:", len(valid_dataset))

    model, optimizer, criterion = initialize_model()

    model = train_model(
        model,
        optimizer,
        criterion,
        device,
        train_dataset,
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        num_epochs=num_epochs,
        verbose=3,
        save=save,
        model_name=model_name,
        save_best=save_best,
    )

    vendors_names = list(metadata.Vendor.unique())
    vendor_datasets_3D = []
    for vendor in vendors_names:
        vendor_datasets_3D.append(load_vendor_3D(vendor, metadata))
    save_results(model_name, device, metadata, vendor_datasets_3D, show_example=True)

    model, optimizer = load_model(model_name, show_performance=True)

    grouped_by_vendor = show_results(model_name)

    return 0


def main():
    return 0


if __name__ == "__main__":
    main()
