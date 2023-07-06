
import glob


def get_subjects_names(dir, root_dir="", verbose=False):
  subjects_names = glob.glob("*", root_dir=root_dir+dir)
  if verbose:
    print(f"subjects for {dir[:-1]}: {len(subjects_names)}")
  return subjects_names



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