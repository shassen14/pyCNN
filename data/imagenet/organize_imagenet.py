import os
from torchvision import datasets

dir_path = os.path.dirname(os.path.realpath(__file__))

datasets.ImageNet(dir_path, "train")
datasets.ImageNet(dir_path, "val")