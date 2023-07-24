from glob import glob
import os

def get_subject_ids(centre, metadata):
    return list(metadata["External code"][metadata.Centre == centre])

def get_subjects_files_paths(centre, root_directory, metadata):
    subject_images = []
    subject_labels = []
    subject_ids = get_subject_ids(centre, metadata)
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
if __name__=="__main__":
    main()