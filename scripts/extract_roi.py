from tqdm import tqdm
import torchio as tio
import os
from scripts.utils import load_metadata, get_subjects_files_paths
from scripts.data import get_channel_index


def get_boundaries(data, heigth, width, padding_size):
    nonzero_indices = data.nonzero(as_tuple=False)
    min_boundaries = nonzero_indices.min(axis=0).values[1:-1]
    max_boundaries = nonzero_indices.max(axis=0).values[1:-1]

    h_init = max(min_boundaries[0].item() - padding_size, 0)
    w_init = max(min_boundaries[1].item() - padding_size, 0)
    h_fin = max(heigth - max_boundaries[0].item() - padding_size, 0)
    w_fin = max(width - max_boundaries[1].item() - padding_size, 0)

    return h_init, h_fin, w_init, w_fin


def extract_ROI(
    data_dir, destination_dir, metadata_path=None, crop_size=128, padding_size=20
):
    metadata = load_metadata(metadata_path)
    subjects_ids = metadata["External code"]

    subjects_ids, subject_images, subject_labels = get_subjects_files_paths(
        data_dir, subject_ids=subjects_ids
    )
    print("Start")
    for subject_id, image_path, seg_path in tqdm(
        zip(subjects_ids, subject_images, subject_labels)
    ):
        try:
            ed_index = get_channel_index("ED", subject_id)
            es_index = get_channel_index("ES", subject_id)

            image = tio.ScalarImage(image_path)
            seg = tio.LabelMap(seg_path)

            # Only keep ED and ES
            image.set_data(image.data[[ed_index, es_index]])
            seg.set_data(seg.data[[ed_index, es_index]])

            c, h, w, d = seg.shape
            h_init, h_fin, w_init, w_fin = get_boundaries(seg.data, h, w, padding_size)

            transform = tio.transforms.Compose(
                [
                    tio.transforms.Crop((h_init, h_fin, w_init, w_fin, 0, 0)),
                    tio.transforms.Resize((crop_size, crop_size, d)),
                ]
            )

            image = transform(image)
            seg = transform(seg)

            subject_dir = os.path.join(destination_dir, str(subject_id))
            os.makedirs(subject_dir, exist_ok=True)

            image.save(os.path.join(subject_dir, f"{subject_id}_sa.nii.gz"))
            seg.save(os.path.join(subject_dir, f"{subject_id}_sa_gt.nii.gz"))

        except:
            print(subject_id, " cannot be loaded")
            continue
    print("Complete")


def main():
    return 0


if __name__ == "__main__":
    main()
