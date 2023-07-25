from glob import glob
import os
import pandas as pd


def load_metadata(path=None):
    if path is None:
        path = "Data/M&Ms/OpenDataset/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"
    metadata = pd.read_csv(path, index_col=0)
    return metadata


def save_metadata(metadata, path=None):
    if path is None:
        path = "Data/M&Ms/OpenDataset/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"
    metadata.to_csv(path)


def get_subjects_files_paths(centre=None, n_max=None):
    metadata = load_metadata()

    if centre:
        metadata = metadata[metadata.Centre == centre]

    if n_max:
        if n_max < len(metadata):
            metadata = metadata[:n_max]

    return metadata[["Image_path", "Seg_path"]]


def main():
    return 0


if __name__ == "__main__":
    main()
