import os
import glob
from tqdm import tqdm
import torchio as tio
from torch.utils.data import Dataset


def load_subject(subject_id, data_dir, folder, transform):
    image_path = os.path.join(data_dir, folder, subject_id, f"{subject_id}_sa.nii.gz")
    seg_path = os.path.join(data_dir, folder, subject_id, f"{subject_id}_sa_gt.nii.gz")

    image = tio.ScalarImage(image_path)
    seg = tio.LabelMap(seg_path)

    subject = tio.Subject(image=image, seg=seg, id=subject_id)

    subject_transformed = transform(subject)
    return subject_transformed


def get_subjects_dir(data_dir):
    subjects_folders = glob.glob("*/*/", root_dir=data_dir)
    subject_ids = [s[-7:-1] for s in subjects_folders]
    return subject_ids, subjects_folders


def create_subject(image, seg, aff):
    image = tio.ScalarImage(tensor=image, aff=aff)
    seg = tio.LabelMap(tensor=seg, aff=aff)

    subject = tio.Subject(image=image, seg=seg)
    return subject


def load_vendor_2D(vendor, metadata, transform=None):
    if transform is None:
        transform = tio.RescaleIntensity((0, 1))
    data_dir = "Data/M&Ms/OpenDataset/"
    subject_ids, subjects_folders = get_subjects_dir(data_dir)

    subjects = []
    for subject_id, folder in tqdm(zip(subject_ids, subjects_folders)):
        if metadata.loc[subject_id].Vendor == vendor:
            subject = load_subject(subject_id, data_dir, folder[:-8], transform)

            image = subject.image.data
            seg = subject.seg.data
            aff = subject.image.affine

            for slice in range(image.shape[-1]):
                ed_slice = image.data[0, :, :, slice].unsqueeze(0).unsqueeze(-1)
                ed_seg = seg.data[0, :, :, slice].unsqueeze(0).unsqueeze(-1)
                subject_ed = create_subject(ed_slice, ed_seg, aff)

                es_slice = image.data[1, :, :, slice].unsqueeze(0).unsqueeze(-1)
                es_seg = seg.data[1, :, :, slice].unsqueeze(0).unsqueeze(-1)
                subject_es = create_subject(es_slice, es_seg, aff)

                subjects.extend((subject_ed, subject_es))

    dataset = tio.SubjectsDataset(subjects)
    return dataset


class VendorDataset(Dataset):
    def __init__(
        self, vendor, metadata, load_transform=None, augmentation_transform=None
    ):
        if load_transform is None:
            load_transform = tio.RescaleIntensity((0, 1))
        self.dataset_2D = load_vendor_2D(vendor, metadata, load_transform)
        self.vendor = vendor
        self.augmentation_transform = augmentation_transform

    def __len__(self):
        return len(self.dataset_2D)

    def __getitem__(self, index):
        subject = self.dataset_2D[index]

        if self.augmentation_transform:
            subject = self.augmentation_transform(subject)

        image = subject.image.data.squeeze().unsqueeze(0)
        label = subject.seg.data.squeeze().unsqueeze(0)

        return image, label
