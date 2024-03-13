from torchvision import datasets

data_dir = "./data/"
train_dir = "train"
val_dir = "val"


def get_datasets(
    dataset_name: str,
) -> {datasets.DatasetFolder, datasets.DatasetFolder}:
    if dataset_name == "tiny-imagenet-200":
        train_path = data_dir + dataset_name + "/" + train_dir
        val_path = data_dir + dataset_name + "/" + val_dir
        train_dataset = datasets.ImageFolder(train_path)
        val_dataset = datasets.ImageFolder(val_path)
    else:
        raise Exception(
            "Dataset name is rather invalid or does not exist. Please correct."
        )

    return {train_dataset, val_dataset}


if __name__ == "__main__":
    train_dataset, val_dataset = get_datasets("tiny-imagenet-200")

    print(len(train_dataset.classes))
