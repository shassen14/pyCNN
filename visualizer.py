import config as cfg
import utils

import sys
import argparse
import PIL.Image as Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

# globals
start_epoch = 1
best_val_loss = sys.float_info.max

# obtain config values
# TODO: figure out how to do global directory instead of relative. hardcoded for now
config = cfg.Config
data_dir = utils.get_folder_path("data")
pt_path = utils.get_file_path(config.param_dir, config.pt_file)

# obtain torch model saved
torch_model = torch.load(pt_path)

# obtain cfg in torch model
# TODO: there probably is some bug by doing it this way. Need to figure out
# a better way to load config that reduces a bug happening from
# differentiating config parameters
config = torch_model["config"]

# Create transforms for preprocessing images to input into models
transform_train, transform_val = utils.get_transforms(
    config.model_name, config.dataset_name
)

# Create dataset for both training and validation
train_dataset, val_dataset = utils.get_datasets(data_dir, config.dataset_name)
train_dataset.transform = transform_train
val_dataset.transform = transform_val

val_loader = DataLoader(
    val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4
)

# obtain model and optimizer
model = utils.get_model(config.model_name, train_dataset, config)
model.load_state_dict(torch_model["model"])
# model.to(config.device_type)
model.eval()

# we will save the conv layer weights in this list
model_weights = []
# we will save the 49 conv layers in this list
conv_layers = []

model_children = list(model.children())
# print(len(model_children))

# counter to keep count of the conv layers
counter = 0
# append all the conv layers and their respective wights to the list
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter += 1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for child in model_children[i].children():
            if type(child) == nn.Conv2d:
                counter += 1
                model_weights.append(child.weight)
                conv_layers.append(child)
            if type(child) == nn.MaxPool2d:
                conv_layers.append(child)
print(f"Total convolution layers: {counter}")

for val_images, val_labels in val_loader:
    sample_image = val_images[0]  # Reshape them according to your needs.
    sample_label = val_labels[0]

    # TODO: make this work
    # pred, correct = utils.model_validate(
    #     model, sample_image, sample_label, config.device_type
    # )

    outputs = [sample_image]
    names = [sample_label]
    for layer in conv_layers[0:]:
        sample_image = layer(sample_image)
        outputs.append(sample_image)
        names.append(str(layer))
    # print feature_maps
    # for feature_map in outputs:
    #     print(feature_map.shape)

    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map, 0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())

    fig = plt.figure(figsize=(30, 50))
    for i in range(len(processed) - 1):
        a = fig.add_subplot(5, 5, i + 1)
        imgplot = plt.imshow(processed[i])
        if i == 0:
            a.set_title("Label = {}".format(int(names[i])), fontsize=30)
            plt.xlabel("Prediction: {}".format("nothing"), fontsize=30)
        else:
            a.set_title(names[i].split("(")[0], fontsize=30)
    plt.savefig(str("feature_maps.jpg"), bbox_inches="tight")
