import config as cfg
import utils

import sys
import math
import argparse
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
pt_path = utils.get_file_path(config.param_dir, config.pt_file)
data_dir = utils.get_folder_path("data")
parser = argparse.ArgumentParser(description="Model Visualizer")
args = utils.visualizer_arg_parser(parser, config, pt_path, data_dir)

# obtain torch model saved
torch_model = torch.load(args.pt_path)

# obtain cfg in torch model
config = torch_model["config"]

# location to save filters, and feature images
filter_img_path = utils.get_folder_path(
    "assets", "{}_{}".format(config.model_name, config.dataset_name)
)

# Create transforms for preprocessing images to input into models
transform_train, transform_val = utils.get_transforms(
    config.model_name, config.dataset_name
)

# Create dataset for both training and validation
train_dataset, val_dataset = utils.get_datasets(args.data_path, config.dataset_name)
train_dataset.transform = transform_train
val_dataset.transform = transform_val

val_loader = DataLoader(
    val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4
)

# obtain model and optimizer
model = utils.get_model(config.model_name, train_dataset, config)
model.load_state_dict(torch_model["model"])
model.eval()

# TODO: below make them to be functions
# obtain model and layer information
model_weights, layers, layers_names = utils.get_model_children(model)

# plot filters
utils.plot_filters(model_weights, filter_img_path, args.save_figures, args.plot_figures)

# feature maps
inputs, classes = next(iter(val_loader))

# pass the image through all the layers
utils.plot_feature_maps(
    inputs[0],
    layers,
    layers_names,
    filter_img_path,
    args.save_figures,
    args.plot_figures,
)

# show plots if available
if args.plot_figures:
    plt.show()

# for val_images, val_labels in val_loader:
#     sample_image = val_images[0]  # Reshape them according to your needs.
#     sample_label = val_labels[0]

#     # TODO: make this work
#     # pred, correct = utils.model_validate(
#     #     model, sample_image, sample_label, config.device_type
#     # )

#     outputs = [sample_image]
#     names = [sample_label]
#     for layer in conv_layers[0:]:
#         sample_image = layer(sample_image)
#         outputs.append(sample_image)
#         names.append(str(layer))
#     # print feature_maps
#     # for feature_map in outputs:
#     #     print(feature_map.shape)

#     processed = []
#     for feature_map in outputs:
#         feature_map = feature_map.squeeze(0)
#         gray_scale = torch.sum(feature_map, 0)
#         gray_scale = gray_scale / feature_map.shape[0]
#         processed.append(gray_scale.data.cpu().numpy())

#     fig = plt.figure(figsize=(30, 50))
#     for i in range(len(processed) - 1):
#         a = fig.add_subplot(5, 5, i + 1)
#         imgplot = plt.imshow(processed[i])
#         if i == 0:
#             a.set_title("Label = {}".format(int(names[i])), fontsize=30)
#             plt.xlabel("Prediction: {}".format("nothing"), fontsize=30)
#         else:
#             a.set_title(names[i].split("(")[0], fontsize=30)
#     plt.savefig(str("feature_maps.jpg"), bbox_inches="tight")
