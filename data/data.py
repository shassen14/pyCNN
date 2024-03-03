from torchvision import datasets
import os

cur_dir = os.path.dirname(os.path.realpath(__file__))
mnist_train_dataset = datasets.MNIST(cur_dir, train=True, download=True)
mnist_val_dataset = datasets.MNIST(cur_dir, train=False, download=True)

# Create datasets for training & validation, download if necessary
fashion_train_dataset = datasets.FashionMNIST(cur_dir, train=True, download=True)
fashion_val_dataset = datasets.FashionMNIST(cur_dir, train=False, download=True)
