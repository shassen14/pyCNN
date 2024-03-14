import torch
import torch.nn as nn


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
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=4096, out_features=num_classes),
        )
        # TODO: is init_weights necessary?
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
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class VGG16(nn.Module):
    """Same input of 224x224 3 channel image"""

    def __init__(self, num_classes: int, dropout: float):
        super().__init__()
        """ Features = layer1, layer2, layer3 """
        # H_out = [H_in + 2p - k]/s + 1 -> floor((224 + 2*1 - 3) / 1) + 1 = 224
        # Convd2d leads to (N, 3, 224, 224) -> (N, 64, 224, 224)
        # H_out = floor((224 + 2*0 - 2)/2 + 1) = 112
        # MaxPool2d lead to (N, 64, 224, 224) -> (N, 64, 112, 112)
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # H_out = floor((112 + 2*1 - 3) / 1) + 1 = 112
        # Convd2d leads to (N, 64, 112, 112) -> (N, 128, 112, 112)
        # H_out = floor((112 + 2*0 - 2)/2 + 1) = 56
        # MaxPool2d lead to (N, 128, 112, 112) -> (N, 128, 56, 56)
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # H_out = floor((56 + 2*1 - 3) / 1) + 1 = 56
        # Convd2d leads to (N, 128, 56, 56) -> (N, 256, 56, 56)
        # H_out = floor((56 + 2*0 - 2)/2 + 1) = 28
        # MaxPool2d lead to (N, 256, 56, 56) -> (N, 256, 28, 28)
        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # H_out = floor((56 + 2*1 - 3) / 1) + 1 = 56
        # Convd2d leads to (N, 256, 28, 28) -> (N, 512, 28, 28)
        # H_out = floor((28 + 2*0 - 2)/2 + 1) = 14
        # MaxPool2d lead to (N, 512, 28, 28) -> (N, 512, 14, 14)
        self.layer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # H_out = floor((28 + 2*1 - 3) / 1) + 1 = 14
        # Convd2d leads to (N, 512, 14, 14) -> (N, 512, 14, 14)
        # H_out = floor((14 + 2*0 - 2)/2 + 1) = 7
        # MaxPool2d lead to (N, 512, 14, 14) -> (N, 512, 7, 7)
        self.layer5 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        """ Classifier is this fully connected neural network which has 3 linear layers"""
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=4096, out_features=num_classes),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
