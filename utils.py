import os
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import config as cfg


###############################################################################
def get_folder_path(folder: str) -> str:
    """Obtain folder path. Side effect is creating one if it doesn't exist"""
    folder_path = os.path.join(os.path.dirname(__file__), folder)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


###############################################################################
def get_file_path(folder: str, file: str) -> str:
    """Obtain file path. Side effect is creating directory if it doesn't exist"""
    file_path = os.path.join(get_folder_path(folder), file)
    return file_path


###############################################################################
def get_datasets(
    data_dir: str, dataset_name: str, train_dir: str = "train", val_dir: str = "val"
) -> {datasets.DatasetFolder, datasets.DatasetFolder}:
    """TODO: add descriptions"""
    dataset_path = os.path.join(data_dir, dataset_name)
    train_path = os.path.join(dataset_path, train_dir)
    val_path = os.path.join(dataset_path, val_dir)

    # TODO: download if not there?
    if dataset_name == "tiny-imagenet-200":
        train_dataset = datasets.ImageFolder(train_path)
        val_dataset = datasets.ImageFolder(val_path)
    elif dataset_name == "imagenet":
        train_dataset = datasets.ImageFolder(train_path)
        val_dataset = datasets.ImageFolder(val_path)
    elif dataset_name == "MNIST":
        train_dataset = datasets.MNIST(data_dir, train=True, download=True)
        val_dataset = datasets.MNIST(data_dir, train=False, download=True)
    elif dataset_name == "FashionMNIST":
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True)
        val_dataset = datasets.FashionMNIST(data_dir, train=False, download=True)
    else:
        raise Exception(
            "Dataset name, " + dataset_name + ", is not supported. Please correct."
        )
    print("Utilizing {} dataset.".format(dataset_name))
    return train_dataset, val_dataset


###############################################################################
# TODO: figure out a good way to input some values in for normalization, etc. Hardcoded transforms for now
def get_transforms(
    model_name: str, dataset_name: str
) -> {transforms.Compose, transforms.Compose}:
    """TODO: add descriptions"""
    if model_name == "AlexNet" or model_name == "VGG16":
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        transform_val = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
        if dataset_name == "MNIST" or dataset_name == "FashionMNIST":
            transform_train = transforms.Compose(
                [transforms.Grayscale(3), transform_train]
            )
            transform_val = transforms.Compose([transforms.Grayscale(3), transform_val])
    else:
        raise Exception("Model, " + model_name + ", is not supported.")
    return transform_train, transform_val


###############################################################################
# TODO: figure out a good way to input values for each model besides from the config
def get_model(
    model_name: str, dataset: datasets.DatasetFolder, config: cfg.Config
) -> {nn.Module}:
    """TODO: add descriptions"""
    # fmt: off
    if model_name == "AlexNet":
        from models.classifiers import AlexNet
        model = AlexNet(len(dataset.classes), dropout=config.dropout)
    elif model_name == "VGG16":
        from models.classifiers import VGG16
        model = VGG16(len(dataset.classes), dropout=config.dropout)
    else:        
        raise Exception(
            "Model, " + model_name + ", is not supported."
        )
    # fmt: on
    print("Utilizing {} model.".format(model_name))
    return model


###############################################################################
def train(model, train_loader, optimizer, criterion, epoch, device):
    model.train()
    input_total = 0
    for batch_idx, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0 or batch_idx == len(train_loader) - 1:
            _, preds = torch.max(output, 1)
            accuracy = torch.sum(preds == labels)
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3f}\tAccuracy: {:.3f}%".format(
                    epoch,
                    input_total,
                    len(train_loader.dataset),
                    100 * input_total / len(train_loader.dataset),
                    loss.item(),
                    100 * accuracy / len(inputs),
                )
            )
        input_total += len(inputs)


###############################################################################
def validate(model, val_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            test_loss += criterion(output, labels).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset) / val_loader.batch_size
    accuracy = 100.0 * correct / len(val_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
            test_loss, correct, len(val_loader.dataset), accuracy
        )
    )
