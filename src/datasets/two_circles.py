from types import SimpleNamespace

import numpy as np
import torch
from sklearn.datasets import make_circles
from torch.utils.data import Dataset


class TwoCirclesDataset(Dataset):
    name = "two_circles"
    input_shape = (2,)
    num_classes = 2

    def __init__(self, n_samples, noise, n_circles=16, factor=0.5, imbalance_ratio=0.0, random_state=None):
        grid_width = np.ceil(np.sqrt(n_circles))

        n_samples = int(n_samples / n_circles)
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)

        for circle_ix in range(1, n_circles):
            new_X, new_y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)
            new_X[:, 0] += 2.5 * (circle_ix % grid_width)
            new_X[:, 1] += 2.5 * (circle_ix // grid_width)

            X = np.concatenate((X, new_X), axis=0)
            y = np.concatenate((y, new_y), axis=0)

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

        X = (X - X.mean(axis=0)) / X.std(axis=0)

        self.data = torch.tensor(X).to(torch.float32)
        self.targets = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


def load_two_circles_dataset(
    train_samples=1000, val_samples=1000, train_noise=0.05, val_noise=0.05, factor=0.5, imbalance_ratio=0.0, **kwargs
) -> SimpleNamespace:
    train_dataset = TwoCirclesDataset(
        train_samples, random_state=0, noise=train_noise, factor=factor, imbalance_ratio=imbalance_ratio
    )

    # Use balanced sampling for the val set
    val_dataset = TwoCirclesDataset(val_samples, random_state=1, noise=val_noise, factor=factor, imbalance_ratio=0.0)

    return SimpleNamespace(train=train_dataset, val=val_dataset), train_dataset.num_classes
