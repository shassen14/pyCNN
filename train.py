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
# TODO: figure out how to do global directory instead of relative. hardcoded for now
config = cfg.Config
data_dir = utils.get_folder_path("data")
pt_path = utils.get_file_path(config.param_dir, config.pt_file)

# Create transforms for preprocessing images to input into models
transform_train, transform_val = utils.get_transforms(
    config.model_name, config.dataset_name
)
print((transform_val))

# Create dataset for both training and validation
train_dataset, val_dataset = utils.get_datasets(data_dir, config.dataset_name)
train_dataset.transform = transform_train
val_dataset.transform = transform_val

# dataloaders
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4
)

# model
model = utils.get_model(config.model_name, train_dataset, config)
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
