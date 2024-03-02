"""
Choose configuration to use
"""

from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class Config:
    # fmt: off
    # Initialize
    initialize: str         = "start"

    # Dataset to utilize

    dataset_dir: str        = "data/imagenet/"
    train_dir: str          = "train"
    val_dir: str            = "val"
    dir_array: List[str]   = field(default_factory=lambda: [Config.train_dir, Config.val_dir])

    # Parameter Save/Load
    param_dir: str          = "params"
    pt_file: str            = "alexnet_init.pt"

    ############################## Paramaters #####################################
    # Model
    classes: int            = 1000
    dropout: float          = 0.2

    # Optimizer
    learning_rate: float    = 0.0001

    # training
    epochs: int             = 6
    batch_size: int         = 128

    # get device type. get GPU or apple if possible
    device_type: str        = "cpu"
    # fmt: on
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"

    print("Device Type: " + device_type)
