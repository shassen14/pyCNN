"""
Choose configuration to use
"""

from dataclasses import dataclass
import torch


@dataclass
class Config:
    # fmt: off
    # Initialize
    initialize: str         = "start"

    # Dataset to utilize
    dataset_name: str       = "tiny-imagenet-200"

    # Model to utilize
    model_name: str         = "AlexNet"

    # Parameter Save/Load
    param_dir: str          = "params/"
    pt_file: str            = "alexnet_init.pt"

    ############################## Paramaters #####################################
    # Model
    dropout: float          = 0.5

    # Optimizer
    learning_rate: float    = 0.0001
    step_size: int          = 2
    gamma: float            = 0.5

    # Training
    epochs: int             = 100
    batch_size: int         = 128

    # get device type. get GPU or apple if possible
    device_type: str        = "cpu"
    # fmt: on
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"

    print("Device Type: " + device_type)
