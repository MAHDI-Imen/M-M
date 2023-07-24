from glob import glob
import os
import pandas as pd


def load_metadata(path=None):
    if path is None:
        path = "Data/M&Ms/OpenDataset/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"
    metadata = pd.read_csv(path, index_col=0)
    return metadata


def get_subject_ids(centre, metadata):
    if metadata is None:
        metadata = load_metadata()
    return list(metadata["External code"][metadata.Centre == centre])


def get_subjects_files_paths(
    root_directory, metadata=None, subject_ids=None, centre=None
):
    if metadata is None:
        metadata = load_metadata()

    if subject_ids is None and centre is None:
        raise ValueError(
            "Either center or subject ids must be given. Both are set to None."
        )

    if centre:
        subject_ids = get_subject_ids(centre, metadata)

    subject_labels = []
    subject_images = []
    subject_ids_final = []

    for subject_id in subject_ids:
        pattern = os.path.join(root_directory, "**", subject_id, "*.nii.gz")
        files_found = glob(pattern, recursive=True)
        if not files_found:
            continue

        files_found.sort()

        image_file, label_file = files_found

        subject_images.append(image_file)
        subject_labels.append(label_file)
        subject_ids_final.append(subject_id)

    return subject_ids_final, subject_images, subject_labels


def main():
    return 0


if __name__ == "__main__":
    main()
