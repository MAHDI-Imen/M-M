import os
from tqdm import tqdm
import torchio as tio
import torch
import pandas as pd
from torch.utils.data import  TensorDataset

import glob


def get_subjects_names(dir, root_dir="", verbose=False):
  subjects_names = glob.glob("*", root_dir=root_dir+dir)
  if verbose:
    print(f"subjects for {dir[:-1]}: {len(subjects_names)}")
  return subjects_names


def load_2D(data_dir, transform=None):
    if transform is None:
        transform = tio.RescaleIntensity((0, 1))

    root_dir = "/home/ids/mahdi-22/M-M/Data/M&Ms/OpenDataset/"
    subjects_ids = get_subjects_names(data_dir, root_dir, verbose=False)

    images = []
    labels = []
    for subject_id in tqdm(subjects_ids):
        image_path = os.path.join(root_dir, data_dir, subject_id, f"{subject_id}_sa.nii.gz")
        seg_path = os.path.join(root_dir, data_dir, subject_id, f"{subject_id}_sa_gt.nii.gz")

        image = tio.ScalarImage(image_path)
        seg = tio.LabelMap(seg_path)

        image = transform(image)
        seg = transform(seg)

        for slice in range(image.shape[-1]):
            ed_slice = image.data[0, :, :, slice].unsqueeze(0)
            images.append(ed_slice)
            es_slice = image.data[1, :, :, slice].unsqueeze(0)
            images.append(es_slice)
            ed_seg = seg.data[0, :, :, slice].unsqueeze(0)
            labels.append(ed_seg)
            es_seg = seg.data[1, :, :, slice].unsqueeze(0)
            labels.append(es_seg)

    dataset = TensorDataset(torch.stack(images), torch.stack(labels))
    n = len(images)

    print(data_dir, 'Dataset size:', n, 'subjects')
    return dataset




def load_3D(data_dir, transform=None):
    if transform==None:
        transform = tio.RescaleIntensity((0,1))

    root_dir = "/home/ids/mahdi-22/M-M/Data/M&Ms/OpenDataset/"
    subjects_ids = get_subjects_names(data_dir, root_dir,verbose=False)


    subjects = []
    for subject_id in tqdm(subjects_ids):

        image_path = os.path.join(root_dir, data_dir, subject_id, f"{subject_id}_sa.nii.gz")
        seg_path = os.path.join(root_dir, data_dir, subject_id, f"{subject_id}_sa_gt.nii.gz")

        image = tio.ScalarImage(image_path)
        seg = tio.LabelMap(seg_path)

        subject = tio.Subject(
            image = image,
            seg = seg,
            id = subject_id
        )

        subjects.append(transform(subject))
    dataset = tio.SubjectsDataset(subjects)
    n = len(dataset)
       
    print(data_dir ,'Dataset size:', n, 'subjects')
    return dataset



def main():
    return 0

if __name__=='__main__':
    main()