import argparse
from tqdm import tqdm
import torchio as tio 
import pandas as pd

from scripts.load_data import get_subjects_names


def get_boundaries(data, width, heigth, padding_size):
    nonzero_indices = data.nonzero(as_tuple=False)
    min_boundaries = nonzero_indices.min(axis=0).values[1:-1]
    max_boundaries = nonzero_indices.max(axis=0).values[1:-1]

    w_init = max(min_boundaries[0].item() - padding_size, 0)
    h_init = max(min_boundaries[1].item() - padding_size, 0)
    w_fin = max(width - max_boundaries[0].item() - padding_size, 0)
    h_fin = max(heigth - max_boundaries[1].item() - padding_size, 0)
    
    return w_init, w_fin, h_init, h_fin


def extract_ROI(data_dir, destination_dir, metadata_path, crop_size=128 ,padding_size=50):


  metadata = pd.read_csv(metadata_path, index_col=1).drop(columns="Unnamed: 0")
  subjects_ids = get_subjects_names(data_dir, verbose=False)
  print("Start")
  for subject_id in tqdm(subjects_ids):
    try:
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
      

      image.save(f"{destination_dir}{subject_id}/{subject_id}_sa.nii.gz")
      seg.save(f"{destination_dir}{subject_id}/{subject_id}_sa_gt.nii.gz")
    except IndexError:
      print(subject_id, " cannot be loaded")
      continue
  print("Complete") 


def main():
  return 0


if __name__ == '__main__':
  main()