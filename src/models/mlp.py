from typing import Type

import torch.nn as nn

from .base_model import BaseModel


class MLP(BaseModel):
    def __init__(
        self, input_size: int, output_size: int, hidden_sizes: tuple = (), activation_type: Type[nn.Module] = nn.ReLU
    ):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes

        layers = []

        # Add hidden layers
        in_size = input_size
        for out_size in hidden_sizes:
            layers.append(nn.Linear(in_size, out_size))
            layers.append(activation_type())
            in_size = out_size

        # Add output layer
        layers.append(nn.Linear(in_size, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.model(x)
        return x


def LogisticRegression(input_size: int, output_size: int):
    return MLP(input_size, output_size, hidden_sizes=())
