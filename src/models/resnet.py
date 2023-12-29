import torch
import torch.nn as nn

from .base_model import BaseModel


class ResNet(BaseModel):
    def __init__(self, num_classes=10, resnet_name="resnet18", weights=None):
        super(ResNet, self).__init__()
        self.resnet = torch.hub.load("pytorch/vision:v0.15.1", resnet_name, weights=weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


class ResNet18MNIST(ResNet):
    def __init__(self, weights=None):
        super(ResNet18MNIST, self).__init__(num_classes=10, resnet_name="resnet18", weights=weights)

        # Modifying the first conv layer as per: https://www.kaggle.com/code/jokyeongmin/mnist-resnet18-in-pytorch
        extract_kwargs = ["out_channels", "kernel_size", "stride", "padding", "bias", "dilation", "groups"]
        existing_kwargs = {k: getattr(self.resnet.conv1, k) for k in extract_kwargs}
        self.resnet.conv1 = nn.Conv2d(in_channels=1, **existing_kwargs)

    def forward(self, x):
        return self.resnet(x)


class ResNet18CIFAR(ResNet):
    def __init__(self, num_classes=10, weights=None):
        super(ResNet18CIFAR, self).__init__(num_classes=num_classes, resnet_name="resnet18", weights=weights)
        self.resnet = torch.hub.load("pytorch/vision:v0.15.1", "resnet18", weights=weights)

        # Modifying the first conv layer as per: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
        self.resnet.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        return self.resnet(x)
