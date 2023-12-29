import cooper
import torch
import torch.nn as nn


class MNISTMultiplierModel(cooper.multipliers.ImplicitMultiplier):
    def __init__(self):
        super(MNISTMultiplierModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=3)
        self.fc1 = nn.Linear(in_features=20 * 8 * 8, out_features=100)

        # self.fc1.weight.data *= 0.1
        # self.fc1.bias.data = torch.zeros_like(self.fc1.bias.data)

        self.fc2 = nn.Linear(in_features=100, out_features=1)
        # self.fc2.weight.data *= 0.1
        # self.fc2.weight.data = torch.randn_like(self.fc2.weight.data) * 1
        # self.fc2.bias.data = torch.zeros_like(self.fc2.bias.data)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, self.fc1.in_features)
        x = torch.tanh(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # x = torch.relu(x)  # Ensure non-negativity of the multiplier output

        return x
