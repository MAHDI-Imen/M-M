import random


# We take a percentage of the data for training:
def split_training_data(metadata, train_ratio=0.8, vendor="A", save=False, seed=42):
    random.seed(seed)
    n_total = metadata["Vendor"].value_counts()[vendor]
    n_train = int(n_total * train_ratio)
    indices = metadata.index[metadata["Vendor"] == vendor].tolist()
    train_indices = random.sample(indices, n_train)

    metadata.loc[train_indices, "Vendor"] = "F"
    metadata.loc[train_indices, "Centre"] = 6

    print(
        f"total number of samples: {n_total}, train samples: {n_train}, Validation: {n_total-n_train}"
    )
    if save:
        metadata_path = "Data/M&Ms/OpenDataset/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"
        metadata.to_csv(metadata_path, index=False)

    return metadata


def main():
    return 0


if __name__ == "__main__":
    main()
