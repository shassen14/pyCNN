from torchvision import datasets
import os

cur_dir = os.path.dirname(os.path.realpath(__file__))
train_dataset = datasets.MNIST(cur_dir, train=True, download=True)
val_dataset = datasets.MNIST(cur_dir, train=False, download=True)
