from torchio.data.dataset import SubjectsDataset


def humanbytes(B):
    """Return the given bytes as a human friendly KB, MB, GB, or TB string."""
    B = float(B)
    KB = float(1024)
    MB = float(KB**2)  # 1,048,576
    GB = float(KB**3)  # 1,073,741,824
    TB = float(KB**4)  # 1,099,511,627,776

    if B < KB:
        return "{0} {1}".format(B, "Bytes" if 0 == B > 1 else "Byte")
    elif KB <= B < MB:
        return "{0:.2f} KB".format(B / KB)
    elif MB <= B < GB:
        return "{0:.2f} MB".format(B / MB)
    elif GB <= B < TB:
        return "{0:.2f} GB".format(B / GB)
    elif TB <= B:
        return "{0:.2f} TB".format(B / TB)


def get_total_memory(dataset, image_types=["image", "seg"]):
    if isinstance(dataset, SubjectsDataset):
        memory_usage = [
            sum([subject[image].memory for image in image_types])
            for subject in dataset.dry_iter()
        ]
        total = sum(memory_usage)
        return humanbytes(total)
    else:
        print("Memory function not implemented for 2D")


def main():
    return 0


if __name__ == "__main__":
    main()
