import config as cfg
from models.alexnet import AlexNet

from torchvision import datasets, transforms
import torch
import torch.nn as nn

# TODO: delete later
import torch.utils.data
import cv2
from data import data


#### functions ####
def train(model, train_loader, optimizer, criterion, epoch, device):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            _, preds = torch.max(output, 1)
            accuracy = torch.sum(preds == labels)
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.3f}%".format(
                    epoch,
                    batch_idx * len(inputs),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                    100 * accuracy / len(inputs),
                )
            )


def validate(model, val_loader, device):
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

    test_loss /= len(val_loader.dataset)
    accuracy = 100.0 * correct / len(val_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss, correct, len(val_loader.dataset), accuracy
        )
    )


if __name__ == "__main__":
    # obtain config values
    config = cfg.Config

    # Create transforms for preprocessing images to input into models
    # TODO: might be good to have these things in configs to change between models
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    transform_train = transforms.Compose(
        [
            transforms.Grayscale(3),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    transform_val = transforms.Compose(
        [
            transforms.Grayscale(3),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Create dataset for both training and validation
    train_dir = config.dataset_dir + config.train_dir
    train_dataset = data.train_dataset
    train_dataset.transform = transform_train

    val_dir = config.dataset_dir + config.val_dir
    val_dataset = data.val_dataset
    val_dataset.transform = transform_val

    # dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4
    )

    # model
    model = AlexNet(len(train_dataset.classes), config.dropout)
    model.to(config.device_type)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    # loss criterion
    criterion = nn.CrossEntropyLoss().to(config.device_type)

    # training
    for epoch in range(1, config.epochs + 1):
        train(model, train_loader, optimizer, criterion, epoch, config.device_type)
        validate(model, val_loader, config.device_type)
        lr_scheduler.step()
        torch.save(model.state_dict(), config.pt_file)
        print("Model saved")
