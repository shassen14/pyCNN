import torch
import torch.nn as nn
from torch.nn import functional as F


class AlexNet(nn.Module):
    def __init__(self, num_classes, dropout):
        """N is a batch size, C denotes a number of channels,
        H is a height of input planes in pixels, and W is width in pixels.
        (N, C_in, H_in, W_in) -> (N, C_out, H_out, W_out)
        Utilizing https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html to understand
        the mathematics in determining H_out, W_out.
        """
        super().__init__()
        """ Features = layer1, layer2, layer3 """
        # H_out = [H_in + 2p - k]/s + 1 -> floor((224 + 2*2 - 11) / 4) + 1 = 55
        # Convd2d leads to (N, 3, 224, 224) -> (N, 96, 55, 55)
        # H_out = floor((55 + 2*0 - 3)/2 + 1) = 27
        # MaxPool2d lead to (N, 96, 55, 55) -> (N, 96, 27, 27)
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # H_out = floor((27 + 2*2 - 5) / 1) + 1 = 27
        # Convd2d leads to (N, 96, 27, 27) -> (N, 256, 27, 27)
        # H_out = floor((27 + 2*0 - 3)/2 + 1) = 13
        # MaxPool2d lead to (N, 256, 27, 27) -> (N, 256, 13, 13)
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # H_out = floor((13 + 2*1 - 3) / 1) + 1 = 13
        # Convd2d leads to (N, 256, 13, 13) -> (N, 384, 13, 13)
        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
        )

        # Convd2d leads to (N, 384, 13, 13) -> (N, 384, 13, 13)
        self.layer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
        )

        # Convd2d leads to (N, 384, 13, 13) -> (N, 256, 13, 13)
        # H_out = floor((13 + 2*0 - 3) / 2) + 1 = 6
        # MaxPool2d lead to (N, 256, 13, 13) -> (N, 256, 6, 6)
        # TODO: Why 256, 6, 6??
        self.layer5 = nn.Sequential(
            nn.Conv2d(
                in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        """ Classifier is this fully connected neural network which has 3 linear layers"""
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),
        )
        self.layer1.apply(self.init_weights)
        self.layer2.apply(self.init_weights)
        self.layer3.apply(self.init_weights)
        self.layer4.apply(self.init_weights)
        self.layer5.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        # print("Input: {}".format(x.shape))
        out = self.layer1(x)
        # print("After Layer 1: {}".format(out.shape))
        out = self.layer2(out)
        # print("After Layer 2: {}".format(out.shape))
        out = self.layer3(out)
        # print("After Layer 3: {}".format(out.shape))
        out = self.layer4(out)
        # print("After Layer 4: {}".format(out.shape))
        out = self.layer5(out)
        # print("After Layer 5: {}".format(out.shape))
        out = torch.flatten(out, 1)
        # print("After Flatten: {}".format(out.shape))
        out = self.fc(out)
        return out
