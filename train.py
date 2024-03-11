import config as cfg
from models import classifiers

from torchvision import datasets, transforms
import torch
import torch.nn as nn
import utils

# TODO: delete later
import torch.utils.data
from data import data


# obtain config values
config = cfg.Config
pt_path = utils.get_file_path(config.param_dir, config.pt_file)

# Create transforms for preprocessing images to input into models
# TODO: might be good to have these things in configs to change between models
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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

# Create dataset for both training and validation
# TODO: figure out obtaining different datasets. Not all are image folders
train_dir = utils.get_folder_path(config.dataset_dir + config.train_dir)
train_dataset = datasets.ImageFolder(train_dir, transform=transform_val)

val_dir = utils.get_folder_path(config.dataset_dir + config.val_dir)
val_dataset = datasets.ImageFolder(val_dir, transform=transform_val)

# dataloaders
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4
)

# model
# TODO: figure out initializing different models
model = classifiers.AlexNet(len(train_dataset.classes), config.dropout)
model.to(config.device_type)

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=config.step_size, gamma=config.gamma
)

# loss criterion
criterion = nn.CrossEntropyLoss().to(config.device_type)

# training
for epoch in range(1, config.epochs + 1):
    utils.train(model, train_loader, optimizer, criterion, epoch, config.device_type)
    utils.validate(model, val_loader, criterion, config.device_type)
    lr_scheduler.step()
    torch.save(model.state_dict(), pt_path)
    print("Model saved")
