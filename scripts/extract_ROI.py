import argparse
from tqdm import tqdm
import torchio as tio 
import pandas as pd

from utils import get_subjects_names


def get_boundaries(data, width, heigth, padding_size):
    nonzero_indices = data.nonzero(as_tuple=False)
    min_boundaries = nonzero_indices.min(axis=0).values[1:-1]
    max_boundaries = nonzero_indices.max(axis=0).values[1:-1]

    w_init = min_boundaries[0].item() - padding_size
    h_init = min_boundaries[1].item() - padding_size
    w_fin = width - max_boundaries[0].item() - padding_size
    h_fin = heigth - max_boundaries[1].item() - padding_size
    
    return w_init, w_fin, h_init, h_fin


def extract_ROI(data_dir, metadata_path, crop_size=128 ,padding_size=10):
  
  metadata = pd.read_csv(metadata_path, index_col=1).drop(columns="Unnamed: 0")
  subjects_ids = get_subjects_names(data_dir, verbose=False)
  print("Start")
  for subject_id in tqdm(subjects_ids):
    ed_index = metadata.loc[subject_id].ED
    es_index = metadata.loc[subject_id].ES


    image = tio.ScalarImage(f"{data_dir}{subject_id}/{subject_id}_sa.nii.gz")
    seg = tio.LabelMap(f"{data_dir}{subject_id}/{subject_id}_sa_gt.nii.gz")

    # Only keep ED and ES
    image.set_data(image.data[[ed_index, es_index]])
    seg.set_data(seg.data[[ed_index, es_index]])

    c, w, h, d = seg.shape
    w_init, w_fin, h_init, h_fin = get_boundaries(seg.data, w, h, padding_size)
    
    transform = tio.transforms.Compose([
        tio.transforms.Crop((w_init, w_fin, h_init, h_fin, 0, 0)),
        tio.transforms.Resize((crop_size, crop_size, d))
        ])
    
    image = transform(image)
    seg = transform(seg)
    

    image.save(f"{data_dir}{subject_id}/{subject_id}_sa.nii.gz")
    seg.save(f"{data_dir}{subject_id}/{subject_id}_sa_gt.nii.gz")

  print("Complete") 


def main():
    parser = argparse.ArgumentParser(description="ROI Extraction")
    parser.add_argument("data_dir", type=str, help="Path to data directory")
    parser.add_argument("metadata_path", type=str, help="Path to metadata file")
    parser.add_argument("-s", "--crop_size", type=int, default=128, help="Final crop size")
    parser.add_argument("-p", "--padding_size", type=int, default=10, help="Padding size")
    args = parser.parse_args()

    extract_ROI(args.data_dir, args.metadata_path, args.crop_size, args.padding_size)


if __name__ == '__main__':
  main()