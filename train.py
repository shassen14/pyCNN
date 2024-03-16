import config as cfg
import utils

import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# globals
start_epoch = 1
best_val_loss = sys.float_info.max

# obtain config values
# TODO: figure out how to do global directory instead of relative. hardcoded for now
config = cfg.Config
data_dir = utils.get_folder_path("data")
pt_path = utils.get_file_path(config.param_dir, config.pt_file)

# command line interface argument parser
parser = argparse.ArgumentParser(description="PyTorch Training")
args = utils.train_arg_parser(parser, config, pt_path)
print(args)

# TODO: probably should just be a boolean. What else would there be besides starting and resuming?
if args.init == "resume":
    print("Resuming a model")
    torch_model = torch.load(pt_path)
    config = torch_model["config"]
    start_epoch = torch_model["epoch"] + 1
    best_val_loss = torch_model["best_val_loss"]
else:
    print("Starting a new model")

# Create transforms for preprocessing images to input into models
transform_train, transform_val = utils.get_transforms(
    config.model_name, config.dataset_name
)

# Create dataset for both training and validation
train_dataset, val_dataset = utils.get_datasets(data_dir, config.dataset_name)
train_dataset.transform = transform_train
val_dataset.transform = transform_val

# dataloaders
train_loader = DataLoader(
    train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4
)
val_loader = DataLoader(
    val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4
)

# model
model = utils.get_model(config.model_name, train_dataset, config)
model.load_state_dict(torch_model["model"]) if args.init == "resume" else None
model.to(config.device_type)

# optimizer
# TODO: possibly overwrite lr, step_size, gamma, etc. in the config object then create the optimizer outside of this if else?
# TODO: add in weight decay for regularization
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
optimizer.load_state_dict(torch_model["optimizer"]) if args.init == "resume" else None
for param_group in optimizer.param_groups:
    param_group["lr"] = args.lr

# TODO: use args/config values
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

# loss criterion
criterion = nn.CrossEntropyLoss().to(config.device_type)

print("Starting Best Val Loss: {:.3E}".format(best_val_loss))

# training
for epoch in range(start_epoch, config.epochs + 1):
    utils.train(model, train_loader, optimizer, criterion, epoch, config.device_type)
    val_loss = utils.validate(model, val_loader, criterion, config.device_type)

    # save torch model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch_model = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "best_val_loss": val_loss,
        }
        torch.save(torch_model, pt_path)
        print("Model saved to {}\n".format(pt_path))
    else:
        scheduler.step()
        print("Reducing learning rate: {:.3E}\n".format(scheduler.get_last_lr()[0]))
