from tqdm.auto import tqdm
import torchio as tio
import os
from scripts.utils import load_metadata, save_metadata


def get_boundaries(data, heigth, width, padding_size):
    """
    Get the boundaries of the ROI. The boundaries are computed
    as a centered box around the segmentation mask mask with a
    symetrical padding of the specified size.

    Parameters
    ----------
    data : torch.Tensor
        Tensor containing the segmentation.
    heigth : int
        Heigth of the segmentation.
    width : int
        Width of the segmentation.
    padding_size : int
        Size of the padding.

    Returns
    -------
    h_init : int
        Start of the ROI in the height dimensio,.
    h_fin : int
        End of the ROI in the height dimension.
    w_init : int
        Start of the ROI in the width dimension.
    w_fin : int
        End of the ROI in the width dimension.

    """

    nonzero_indices = data.nonzero(as_tuple=False)
    min_boundaries = nonzero_indices.min(axis=0).values[1:-1]
    max_boundaries = nonzero_indices.max(axis=0).values[1:-1]

    h_init = max(min_boundaries[0].item() - padding_size, 0)
    w_init = max(min_boundaries[1].item() - padding_size, 0)
    h_fin = max(heigth - max_boundaries[0].item() - padding_size, 0)
    w_fin = max(width - max_boundaries[1].item() - padding_size, 0)

    return h_init, h_fin, w_init, w_fin


def extract_ROI(destination_dir, crop_size=128, padding_size=20):
    """
    Extract the region of interest (ROI) from the images and segmentations
    and save them in the destination_dir.

    Parameters
    ----------
    destination_dir : str
        Path to the directory where the ROI will be saved.
    crop_size : int, optional
        Size of the ROI. The default is 128.
    padding_size : int, optional
        Size of the padding. The default is 20.

    Returns
    -------
    None.
    """
    metadata = load_metadata()
    subjects_ids = list(metadata.index)

    print("Start")
    for subject_id in tqdm(subjects_ids, desc="Extracting ROI:", unit="subject"):
        try:
            ed_index, es_index, image_path, seg_path = metadata.loc[
                subject_id, ["ED", "ES", "Image_path", "Seg_path"]
            ]

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
            print(subject_id, " cannot be loaded and will be removed from metadata")
            metadata = metadata.drop(subject_id)
            save_metadata(metadata)
            continue
    print("Complete")


def main():
    destination_dir = "Data/M&Ms/OpenDataset/"

    extract_ROI(destination_dir, crop_size=128, padding_size=30)


if __name__ == "__main__":
    main()
