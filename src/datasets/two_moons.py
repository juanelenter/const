from types import SimpleNamespace

import numpy as np
import torch
from sklearn.datasets import make_moons
from torch.utils.data import Dataset


class TwoMoonsDataset(Dataset):
    name = "two_moons"
    input_shape = (2,)
    num_classes = 2

    def __init__(self, n_samples, noise, imbalance_ratio=0.0, random_state=None):
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

        if not (0 <= imbalance_ratio <= 1):
            raise ValueError("Imbalance ratio must be between 0 (no imbalance) and 1 (full imbalance).")

        if imbalance_ratio > 0:
            # Calculate the number of samples to remove from the minority class
            num_minority_samples = int(np.sum(y == 1) * imbalance_ratio)
            minority_indices = np.where(y == 1)[0]
            removed_indices = np.random.choice(minority_indices, size=num_minority_samples, replace=False)

            # Remove the minority samples from the dataset
            X = np.delete(X, removed_indices, axis=0)
            y = np.delete(y, removed_indices)

        self.data = torch.tensor(X).to(torch.float32)
        self.targets = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


def load_two_moons_dataset(
    train_samples=1000, val_samples=1000, train_noise=0.1, val_noise=0.1, imbalance_ratio=0.0, **kwargs
) -> SimpleNamespace:
    train_dataset = TwoMoonsDataset(train_samples, random_state=0, noise=train_noise, imbalance_ratio=imbalance_ratio)

    # Use balanced sampling for the validation set
    val_dataset = TwoMoonsDataset(val_samples, random_state=1, noise=val_noise, imbalance_ratio=0.0)

    return SimpleNamespace(train=train_dataset, val=val_dataset), train_dataset.num_classes
