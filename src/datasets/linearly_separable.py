from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import chi2
from torch.utils.data import Dataset

from src.utils.plots import plot_max_margin


def sample_truncated_isotropic_gaussian(num_samples, dim, delta=0.0, seed=None):
    # Ensure the sampled points are not too far from the mean
    # This truncates the Gaussian distribution by removing a fraction delta of the mass
    norm_bound = chi2(dim).ppf(1 - delta)

    rng = np.random.default_rng(seed)

    if delta <= 0.1:
        # If chopped proportion is too small, avoid chi2 numerical instability and don't chop
        Z = rng.standard_normal((num_samples, dim))
    else:
        partial_Z, accum = [], 0
        while accum <= num_samples:
            z = rng.standard_normal((num_samples, dim))
            partial_Z.append(z[np.linalg.norm(z, axis=1) < norm_bound])
            accum += len(partial_Z[-1])

        Z = np.concatenate(partial_Z)[:num_samples, ...]

    return Z


def sample_gaussians_and_make_linearly_separable(mean_diff, L0, L1, n0, n1, margin=1.0, delta=0, sample_seed=0):
    dim = np.shape(L1)[0]

    Z0 = sample_truncated_isotropic_gaussian(n0, dim, delta=delta, seed=sample_seed)
    X0 = Z0 @ L0

    # Use a different seed for sampling Gaussian noise for the second class
    Z1 = sample_truncated_isotropic_gaussian(n1, dim, delta=delta, seed=sample_seed + 1)
    X1 = Z1 @ L1

    # Ensure a minimum margin between the two classes
    # We simply use the mean difference vector to generate a separating hyperplane by
    # shifting the points for the two classes in opposite directions along this vector
    # This constitutes a lower bound on the maximum margin that can be achieved.
    gamma_star = -np.min([np.min(X1 @ mean_diff.T - margin), np.min(-X0 @ mean_diff.T - margin)])
    X1 = X1 + gamma_star * mean_diff
    X0 = X0 - gamma_star * mean_diff

    return X1, X0


class LinearlySeparableDataset(Dataset):
    name = "linearly_separable"
    num_classes = 2

    def __init__(
        self,
        num_samples,
        dim,
        L_tuple=None,
        mean_diff=None,
        margin=1.0,
        delta=0.0,
        positive_proportion=0.5,
        data_seed=0,
        sample_seed=0,
        normalize=True,
    ):
        self.input_shape = (dim,)

        n1 = int(num_samples * positive_proportion)
        n0 = num_samples - n1

        data_rng = np.random.default_rng(data_seed)

        if L_tuple is None:
            # Sample Cholesky factors for the covariance matrices
            # Set random state
            L0 = 2 * (data_rng.random((dim, dim)) - 0.5)
            L1 = 2 * (data_rng.random((dim, dim)) - 0.5)
        else:
            L0, L1 = L_tuple

        if mean_diff is None:
            # Fix distribution parameter with seed
            mean_diff = data_rng.standard_normal((1, dim))

        mean_diff = mean_diff / np.linalg.norm(mean_diff)

        X0, X1 = sample_gaussians_and_make_linearly_separable(
            mean_diff, L1, L0, n1, n0, margin=margin, delta=delta, sample_seed=sample_seed
        )

        X = np.concatenate([X0, X1])
        y = np.array(n0 * [0] + n1 * [1]).reshape(-1)  # Labels are 0/1

        # Normalize the coordinates of the data points so maximum absolute magnitude is 1
        if normalize:
            X /= np.max(np.abs(X))

        self.data = torch.tensor(X).to(torch.float32)
        self.targets = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


def load_linearly_separable_dataset(
    train_samples=1000,
    val_samples=1000,
    dim=2,
    margin=1.0,
    positive_proportion=0.5,
    data_seed=0,
    sample_seed=0,
    **kwargs,
) -> SimpleNamespace:
    train_dataset = LinearlySeparableDataset(
        train_samples,
        dim,
        margin=margin,
        data_seed=data_seed,
        sample_seed=sample_seed,
        positive_proportion=positive_proportion,
    )

    # Use balanced sampling for the validation set
    val_dataset = LinearlySeparableDataset(
        val_samples, dim, margin=margin, data_seed=data_seed, sample_seed=sample_seed + 1, positive_proportion=0.5
    )

    return SimpleNamespace(train=train_dataset, val=val_dataset), train_dataset.num_classes


if __name__ == "__main__":
    dataset = LinearlySeparableDataset(500, 2, margin=1.0, delta=0.0, data_seed=0, sample_seed=0, normalize=True)

    for class_id, c in zip([0, 1], ["red", "blue"]):
        class_filter = (dataset.targets == class_id).reshape(-1)
        plt.scatter(dataset.data[class_filter, 0], dataset.data[class_filter, 1], color=c, label=f"Class {class_id}")

    # Find the maximum margin hyperplane using sklearn (use C=1e8 to approximate hard margin)
    plot_max_margin(dataset.data, dataset.targets, plot_paralles=True)

    plt.show()
