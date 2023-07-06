from tqdm import tqdm
import torchio as tio
import pandas as pd
from scripts.utils import get_subjects_names

def create_slice(image, seg, transform, slice, phase):
  subject = tio.Subject(
    image = tio.ScalarImage(tensor=image.data[phase,:,:,slice].unsqueeze(0).unsqueeze(0)),
    seg = tio.LabelMap(tensor=seg.data[phase,:,:,slice].unsqueeze(0).unsqueeze(0)),
  )
  return transform(subject)

    
def load_data(data_dir, dim=2, metadata_path=None, transform=None):
    if metadata_path==None:
        metadata_path = "/home/ids/mahdi-22/M-M/Data/M&Ms/OpenDataset/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"
    if transform==None:
        transform = tio.RescaleIntensity((0,1))

    
    metadata = pd.read_csv(metadata_path, index_col=1).drop(columns="Unnamed: 0")
    root_dir = "/home/ids/mahdi-22/M-M/Data/M&Ms/OpenDataset/"
    subjects_ids = get_subjects_names(data_dir, root_dir,verbose=False)

    subjects = []
    for subject_id in tqdm(subjects_ids):
        image = tio.ScalarImage(f"{root_dir}{data_dir}{subject_id}/{subject_id}_sa.nii.gz")
        seg = tio.LabelMap(f"{root_dir}{data_dir}{subject_id}/{subject_id}_sa_gt.nii.gz")

        if dim==2:
            for slice in range(image.shape[-1]):
                ed_slice = create_slice(image, seg, transform, slice=slice, phase=0)
                subjects.append(ed_slice)
                es_slice = create_slice(image, seg, transform, slice=slice, phase=1)
                subjects.append(es_slice)
        else:    
            subject = tio.Subject(
              image = image,
              seg = seg,
            )
            subjects.append(transform(subject))

    dataset = tio.SubjectsDataset(subjects)
    print(data_dir ,'Dataset size:', len(dataset), 'subjects')

    return dataset


def main():
    return 0

if __name__=='__main__':
    main()