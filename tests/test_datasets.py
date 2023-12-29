import pytest
import torch

from configs.data import mnist_config
from src import datasets


@pytest.mark.parametrize("noise_map", [None, {0: [1, 2, 3], 8: [4, 6, 8]}])
def test_label_noise(noise_map):
    """Check that the label noise is applied consistently given the same seed."""

    config = mnist_config()
    data_class = datasets.KNOWN_DATASET_LOADERS[config.dataset_name]

    targets = {}
    for ix, seed in enumerate([0, 0, 42]):
        dataset_namespace = data_class(**config.dataset_kwargs)
        datasets.utils.apply_classification_label_noise_(
            dataset_namespace.train, noise_ratio=0.1, noise_map=noise_map, seed=seed
        )
        targets[ix] = dataset_namespace.train.dataset.targets

    torch.testing.assert_close(targets[0], targets[1])
    assert not torch.allclose(targets[0], targets[2])


def test_label_imbalance():
    """Check that the class imbalance is applied consistently given the same seed by
    verifying that the chosen sample indices are the same."""

    config = mnist_config()
    data_class = datasets.KNOWN_DATASET_LOADERS[config.dataset_name]

    # Slight perturbation of the dataset to emphasize 1s and reduce 0s
    class_ratios = {0: 0.05, 1: 0.15, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1, 8: 0.1, 9: 0.1}

    chosen_indices = {}
    for ix, seed in enumerate([0, 0, 42]):
        dataset_namespace = data_class(**config.dataset_kwargs)
        dataset = datasets.utils.apply_classification_label_imbalance(
            dataset_namespace.train, class_ratios=class_ratios, seed=seed
        )
        chosen_indices[ix] = dataset.indices

    assert chosen_indices[0] == chosen_indices[1]
    assert not chosen_indices[0] == chosen_indices[2]
