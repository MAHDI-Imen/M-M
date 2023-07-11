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

def get_subjects_dir(data_dir):
    subjects_folders = glob.glob("*/*/", root_dir=data_dir)
    subject_ids = [s[-7:-1] for s in subjects_folders]
    return subject_ids, subjects_folders
        

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

def load_subject(subject_id, data_dir, folder ,transform):
    image_path = os.path.join(data_dir, folder , subject_id ,f"{subject_id}_sa.nii.gz")
    seg_path = os.path.join(data_dir, folder, subject_id ,f"{subject_id}_sa_gt.nii.gz")

    image = tio.ScalarImage(image_path)
    seg = tio.LabelMap(seg_path)

    subject = tio.Subject(
        image = image,
        seg = seg,
        id = subject_id
    )

    subject_transformed = transform(subject)
    return subject_transformed


def load_vendor_2D(vendor, metadata,transform=None):
    if transform is None:
        transform = tio.RescaleIntensity((0, 1))

    data_dir = "Data/M&Ms/OpenDataset/"
    subject_ids, subjects_folders = get_subjects_dir(data_dir)

    images = []
    labels = []
    for subject_id, folder in tqdm(zip(subject_ids, subjects_folders)):
        if metadata.loc[subject_id].Vendor == vendor:
            subject = load_subject(subject_id, data_dir, folder[:-8] ,transform)

            image = subject.image.data
            seg = subject.seg.data

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


def load_vendor_3D(vendor, metadata, transform=None):
    if transform==None:
        transform = tio.RescaleIntensity((0,1))

    data_dir = "Data/M&Ms/OpenDataset/"
    subject_ids, subjects_folders = get_subjects_dir(data_dir)

    subjects = []
    for subject_id, folder in tqdm(zip(subject_ids, subjects_folders)):
        if metadata.loc[subject_id].Vendor == vendor:
            subject = load_subject(subject_id, data_dir, folder[:-8] ,transform)
            subjects.append(subject)

    dataset = tio.SubjectsDataset(subjects)
    n = len(dataset)
       
    print(data_dir ,'Dataset size:', n, 'subjects')
    return dataset


def main():
    return 0

if __name__=='__main__':
    main()