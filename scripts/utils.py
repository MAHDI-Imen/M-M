
import glob


def get_subjects_names(dir, root_dir="", verbose=False):
  subjects_names = glob.glob("*", root_dir=root_dir+dir)
  if verbose:
    print(f"subjects for {dir[:-1]}: {len(subjects_names)}")
  return subjects_names


def humanbytes(B):
    """Return the given bytes as a human friendly KB, MB, GB, or TB string."""
    B = float(B)
    KB = float(1024)
    MB = float(KB ** 2) # 1,048,576
    GB = float(KB ** 3) # 1,073,741,824
    TB = float(KB ** 4) # 1,099,511,627,776

    if B < KB:
        return '{0} {1}'.format(B,'Bytes' if 0 == B > 1 else 'Byte')
    elif KB <= B < MB:
        return '{0:.2f} KB'.format(B / KB)
    elif MB <= B < GB:
        return '{0:.2f} MB'.format(B / MB)
    elif GB <= B < TB:
        return '{0:.2f} GB'.format(B / GB)
    elif TB <= B:
        return '{0:.2f} TB'.format(B / TB)


def get_total_memory(dataset, image_types=["image", "seg"]):
  memory_usage = [sum([subject[image].memory for image in image_types]) for subject in dataset.dry_iter()]
  total = sum(memory_usage)
  return humanbytes(total)


def main():
 print("This is a library with utils functions")
 print("get_subjects_names(dir, root_dir, verbose=False)")
 root_dir = "/home/ids/mahdi-22/M-M/Data/M&M/OpenDataset/"
 train_subjects = get_subjects_names("Training/Labeled/", root_dir ,verbose=True)
 valid_subjects = get_subjects_names("Validation/", root_dir,verbose=True)
 test_subjects = get_subjects_names("Testing/", root_dir,verbose=True)
 print("Example:")
 print("valid_subjects=", valid_subjects[:5])

if __name__ == '__main__':
  main()