from glob import glob
import os
import pandas as pd
from time import time, strftime, gmtime


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


class Timer:
    def __init__(self, time_format="%H:%M:%S", function_name="run"):
        self.time_format = time_format
        self.function_name = function_name

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            start_time = time()
            func(*args, **kwargs)
            end_time = time()
            elapsed_time = strftime(self.time_format, gmtime(end_time - start_time))
            print(f"Finished {self.function_name} in {elapsed_time}.")

        return wrapper


def get_file_basenames_with_path_format(file_path_format):
    file_paths = list(glob(file_path_format))
    file_basenames = [get_file_basename_from_path(path) for path in file_paths]
    return file_basenames


def get_file_basename_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]


def get_venv_status():
    return os.environ.get("VIRTUAL_ENV")


def main():
    return 0


if __name__ == "__main__":
    main()
