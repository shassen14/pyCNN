import config as cfg
import os
import argparse
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import math


###############################################################################
def get_folder_path(*folders: str) -> str:
    """Obtain folder path. Side effect is creating one if it doesn't exist"""
    folder_path = os.path.dirname(__file__)
    for folder in folders:
        folder_path = os.path.join(folder_path, folder)
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
    return test_loss


###############################################################################
def model_validate(model, input, label, device):
    """TODO: fix this.... fix all of this"""
    model.eval()
    with torch.no_grad():
        input, label = input.to(device), label.to(device)
        output = model(input)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(label.view_as(pred)).sum().item()

    return pred, correct


###############################################################################
def get_model_children(model: nn.Module):
    model_weights = []
    layers = []
    layers_names = []
    model_children = list(model.children())

    # counter to keep count of the conv layers
    conv_counter = 0
    maxpool_counter = 0

    # append all the conv layers and their respective wights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            conv_counter += 1
            model_weights.append(model_children[i].weight)
            layers.append(model_children[i])
            layers_names.append(f"Convolutional Layer {conv_counter}")
        elif type(model_children[i]) == nn.MaxPool2d:
            maxpool_counter += 1
            layers.append(model_children[i])
            layers_names.append(f"Max Pool Layer {conv_counter}")
        elif type(model_children[i]) == nn.Sequential:
            for child in model_children[i].children():
                if type(child) == nn.Conv2d:
                    conv_counter += 1
                    model_weights.append(child.weight)
                    layers.append(child)
                    layers_names.append(f"Convolutional Layer {conv_counter}")
                if type(child) == nn.MaxPool2d:
                    maxpool_counter += 1
                    layers.append(child)
                    layers_names.append(f"Max Pool Layer {conv_counter}")
    print(f"Total convolution layers: {conv_counter}")
    print(f"Total max pool layers: {maxpool_counter}")

    return model_weights, layers, layers_names


###############################################################################
def plot_filters(
    model_weights, filter_img_path: str, save_figures: bool, show_figures: bool
):
    # skip everything if doing nothing
    if not save_figures and not show_figures:
        print("Skipping convolutional layers plotting and saving")
        return

    # plot, save and/or show plots
    for i, model_weight in enumerate(model_weights):
        plt.figure(figsize=(16, 9))
        for j, filter in enumerate(model_weight):
            plt.subplot(
                math.ceil(math.sqrt(len(model_weight))),
                math.ceil(math.sqrt(len(model_weight))),
                j + 1,
            )
            plt.imshow(filter[0, :, :].detach(), cmap="gray")
            plt.axis("off")
        plt.suptitle(
            "Convolution Layer {}: {} filters".format(i + 1, len(model_weight)),
            fontsize=40,
        )
        if save_figures:
            plt.savefig(
                get_file_path(filter_img_path, "conv_layer{}.png".format(i + 1))
            )
    if show_figures:
        plt.show()


###############################################################################
def plot_feature_maps(
    input_img,
    layers,
    filter_img_path: str,
    save_figures: bool,
    show_figures: bool,
):
    # skip everything if doing nothing
    if not save_figures and not show_figures:
        print("Skipping feature maps plotting and saving")
        return

    # plot, save and/or show plots
    results = [input_img]
    for i in range(0, len(layers)):
        # pass the result from the last layer to the next layer
        results.append(layers[i](results[-1]))

    for num_layer in range(len(results)):
        plt.figure(figsize=(30, 30))
        layer_viz = results[num_layer][:, :, :]
        layer_viz = layer_viz.data
        for i, filter in enumerate(layer_viz):
            plt.subplot(
                math.ceil(math.sqrt(len(layer_viz))),
                math.ceil(math.sqrt(len(layer_viz))),
                i + 1,
            )
            plt.suptitle(
                "Layer {}: Feature Maps".format(num_layer),
                fontsize=60,
            )
            plt.imshow(filter, cmap="gray")
            plt.axis("off")
        if save_figures:
            print(f"Saving layer {num_layer} feature maps...")
            plt.savefig(f"{filter_img_path}/feature_map{num_layer}.png")
    if show_figures:
        plt.show()
        plt.close()


###############################################################################
def train_arg_parser(
    parser: argparse.ArgumentParser, config: cfg.Config, pt_path: str
) -> argparse.Namespace:
    """
    args.init
    args.pt
    args.lr
    """
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=config.learning_rate,
        type=float,
        help="optimizer's initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "-pt",
        "--pt_path",
        default=pt_path,
        type=str,
        help="absolute path to the saved pytorch model",
        dest="pt",
    )
    parser.add_argument(
        "-i",
        "--init",
        default=config.initialize,
        type=str,
        help="absolute path to the saved pytorch model",
        dest="init",
    )
    return parser.parse_args()


###############################################################################
def visualizer_arg_parser(
    parser: argparse.ArgumentParser, config: cfg.Config, pt_path: str, data_path: str
) -> argparse.Namespace:
    """
    args.init
    args.pt
    args.lr
    """
    parser.add_argument(
        "-pt",
        "--pt_path",
        default=pt_path,
        type=str,
        help="absolute path to the saved pytorch model",
        dest="pt_path",
    )
    parser.add_argument(
        "-d",
        "--data_path",
        default=data_path,
        type=str,
        help="absolute path to the data directory",
        dest="data_path",
    )
    parser.add_argument(
        "-sf",
        "--save_figures",
        action="store_true",
        help="If argument is included when running, figures will be saved as png to the model's assets directory",
        dest="save_figures",
    )
    parser.add_argument(
        "-pf",
        "--plot_figures",
        action="store_true",
        help="If argument is included when running, figures will be shown duering runtime.",
        dest="plot_figures",
    )
    return parser.parse_args()
