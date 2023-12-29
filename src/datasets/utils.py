import time
from types import SimpleNamespace
from typing import Literal, Optional

import numpy as np
import scipy as sp
import torch
from torch.utils.data import DataLoader, Dataset

import shared
from src.utils.utils import RNGContext

from .prefetch_loader import PrefetchLoader

logger = shared.fetch_main_logger(apply_basic_config=True)


class IndexedDataset(Dataset):
    """Wraps a generic Pytorch dataset and appends the index of the sample as the last
    return value in the `__getitem__` method.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        standard_return = self.dataset.__getitem__(idx)
        if isinstance(standard_return, tuple):
            return (*standard_return, idx)
        else:
            return (standard_return, idx)

    def __len__(self):
        return len(self.dataset)


def apply_classification_label_noise_(
    dataset: Dataset, noise_ratio: float, noise_map: Optional[dict] = None, seed: int = 0
) -> torch.Tensor:
    """Apply label noise in-place to a dataset by randomly flipping the labels of
    noise_ratio (%) of the samples.

    If a noise map is provided, the labels will be flipped according to the map, picking
    a random label from the list of possible corrupted labels. The noise map may contain
    a reference to the original label amongst the options for the corrupted label.

    Otherwise, the labels will be flipped randomly. In this case, the labels for the
    selected samples are guaranteed to be corrupted.

    Args:
        dataset (Dataset): Dataset to be corrupted.
        noise_ratio (float): Ratio of samples to be corrupted.
        noise_map (dict, optional): Dictionary mapping the original labels to the
            corrupted ones. The keys must be the original labels and the values must
            be lists of the corrupted labels. Defaults to None.
        seed (int, optional): Seed for the random number generator. Defaults to 0.

    Example:
        >>> dataset = torchvision.datasets.MNIST(root=".", download=True)
        >>> noise_map = {0: [1, 2, 3], 8: [4, 6, 8]}
        >>> apply_classification_label_noise_(dataset, noise_ratio=0.1, noise_map=noise_map, seed=0)

    """

    logger.info(f"Applying label corruption with noise ratio {noise_ratio} and seed {seed}")

    if not (0 <= noise_ratio <= 1):
        raise ValueError(f"Label noise ratio must be between 0 and 1, got {noise_ratio}")

    targets = dataset.targets

    # Select a subset of the indexes to be corrupted
    num_samples = len(dataset)
    num_corrupted_samples = int(noise_ratio * num_samples)

    if noise_map is not None:
        # Check that all keys in the noise map are unique
        if len(noise_map.keys()) != len(set(noise_map.keys())):
            raise ValueError("Noise map must have unique keys")
    else:
        # If not provided, build a noise map with all possible labels (except the original one)
        noise_map = {}
        unique_labels = torch.unique(targets)
        for label in unique_labels:
            noise_map[label.item()] = unique_labels[unique_labels != label]

    with RNGContext(seed):
        source_corrupted_idxs = torch.randperm(num_samples)
        corrupted_idxs = []

        for idx in source_corrupted_idxs:
            current_label = targets[idx].item()
            if current_label in noise_map:
                targets[idx] = np.random.choice(noise_map[current_label])
                corrupted_idxs.append(idx)

            if len(corrupted_idxs) == num_corrupted_samples:
                break

    if len(corrupted_idxs) != num_corrupted_samples:
        raise RuntimeError(f"Could only corrupt {len(corrupted_idxs)} samples out of {num_corrupted_samples} requested")

    return torch.tensor(corrupted_idxs, dtype=targets.dtype, device=targets.device)


def find_feasible_imbalanced_allocation(
    class_ratios: dict, available_samples_per_class: dict, atol=1e-2
) -> dict[int, int]:
    """Finds the largest dataset that satisfies prescribed class ratios. This is done
    by solving a relaxed continuous linear program within the range of available samples
    for each class. The relaxed solution decimal is then rounded down to the nearest
    integer.

    We aim to solve the following linear program:

    .. math::

        \max_x \ & c^T x \\
        \mbox{such that} \ & a_i / a_j = x_i / j_j, \; \forall i, j \in \{1, \dots, C\} \\
        & 0 \leq x \leq U ,

    Args:
        class_ratios (dict): Dictionary of class ratios.
        available_samples_per_class (dict): Dictionary of available number of samples
            per class.
    """

    ordered_keys = sorted(class_ratios.keys())

    gain_per_class = np.array([class_ratios[k] for k in ordered_keys])
    bounds = [(0, available_samples_per_class[k]) for k in ordered_keys]

    # Populate the equality constraints
    # This aims to find x_i and x_j such that the ratio is preserved: a_i / a_j = x_i / x_j
    row_indices, col_indices, data = [], [], []
    row = 0
    for i in class_ratios.keys():
        for j in class_ratios.keys():
            if i <= j:
                row_indices.append(row)
                col_indices.append(i)
                data.append(class_ratios[j])

                row_indices.append(row)
                col_indices.append(j)
                data.append(-class_ratios[i])

                row += 1

    # Densify a sparse COO matrix
    A = sp.sparse.coo_matrix((data, (row_indices, col_indices))).todense()
    b = np.zeros((A.shape[0]))

    # Need to flip the sign since linprog minimizes by default
    res = sp.optimize.linprog(-gain_per_class, A_eq=A, b_eq=b, bounds=bounds)

    # Round down to the nearest integer
    found_allocation = np.floor(res.x)

    if np.allclose(found_allocation / np.sum(found_allocation), gain_per_class, atol=atol):
        return {k: int(found_allocation[ix]) for ix, k in enumerate(ordered_keys)}
    else:
        raise ValueError("No feasible allocation found to match the desired class ratios.")


def apply_classification_label_imbalance(
    dataset: Dataset,
    class_ratios: dict = {},
    seed: int = 0,
    shuffle: bool = True,
    type: Literal["exp", "step"] = "exp",
    ratio: float = 0.1,
) -> Dataset:
    """Apply label imbalance to a dataset by re-sampling the datapoints according to
    their class so that the ratio of samples per class matches the provided ratio.

    Args:
        dataset (Dataset): Dataset to be corrupted.
        class_ratios (dict): Ratio of samples per class.
        seed (int, optional): Seed for the random number generator. Defaults to 0.
        shuffle (bool, optional): Whether to shuffle the indices before constructing
            the subset dataset. Defaults to True.
        type (str, optional): If class ratios are not specified, how to construct them following CIFAR and ImageNet LT.
                                'exp' for exponential distribution, 'step' for step function. Defaults to 'exp'.
    """

    if len(class_ratios) == 0:
        if 0.0 > ratio or ratio > 1:
            raise ValueError("imbalance_rate must be between 0.0 and 1")
        if type == "exp":
            cls_num = len(torch.unique(torch.tensor(dataset.targets)))
            total_ratios = 0
            for cls_idx in range(cls_num):
                num = ratio ** (cls_idx / (cls_num - 1.0))
                class_ratios[cls_idx] = num
                total_ratios += num
        elif type == "step":
            for cls_idx in range(cls_num // 2):
                class_ratios[cls_idx] = 1.0
                total_ratios += 1.0
            for cls_idx in range(cls_num // 2):
                class_ratios[cls_idx] = ratio
                total_ratios += ratio
        for cls_idx in range(cls_num):
            class_ratios[cls_idx] /= total_ratios

    logger.info(f"Applying label imbalance with ratio per class {class_ratios} and seed {seed}")

    if len(class_ratios.keys()) != len(set(class_ratios.keys())):
        raise ValueError("`class_ratios` must have unique keys")

    if not np.isclose(sum(class_ratios.values()), 1):
        raise ValueError(f"Ratio per class must sum up to 1, got {sum(class_ratios.values())}")

    targets = torch.tensor(dataset.targets)

    counts_per_class = torch.bincount(targets)
    available_samples_per_class = {k: v.item() for k, v in enumerate(counts_per_class)}

    with RNGContext(seed):
        # Find the largest dataset that satisfies the prescribed class ratios
        feasible_allocation = find_feasible_imbalanced_allocation(class_ratios, available_samples_per_class)

        inverse_dict = {label.item(): [] for label in torch.unique(targets)}
        for ix, target in enumerate(targets):
            inverse_dict[target.item()].append(ix)

        sampled_idxs = []
        for k, v in feasible_allocation.items():
            sampled_idxs.extend(np.random.choice(inverse_dict[k], v, replace=False))

        # Shuffle the indices before constructing the subset dataset
        if shuffle:
            np.random.shuffle(sampled_idxs)

    return torch.utils.data.Subset(dataset, sampled_idxs), SimpleNamespace(samples_per_class=feasible_allocation)


def find_best_num_workers(dataloader_lmbda):
    logger.info("Finding best num_workers for dataloader")

    num_workers_to_test = list(range(0, 10, 2))
    num_workers_time = {}
    for num_workers_ix, num_workers in enumerate(num_workers_to_test):
        dataloader = dataloader_lmbda(num_workers)
        start = time.time()
        for epoch in range(2):
            for batch_ix, data in enumerate(dataloader, 0):
                if batch_ix > 5:
                    break
        current_time = time.time() - start
        num_workers_time[num_workers] = current_time
        logger.info(f"num_workers: {num_workers}, time: {current_time}")

        if num_workers_ix > 0:
            previous_time = num_workers_time[num_workers_to_test[num_workers_ix - 1]]
            if current_time > previous_time:
                logger.info("Latest num_workers choice caused a time increase, stopping search")
                break

    # Return key with the best time
    return min(num_workers_time, key=num_workers_time.get)


def create_dataloader_for_dataset(
    dataset: Dataset,
    split: str,
    batch_size_per_gpu: int,
    use_distributed_sampler: bool = False,
    seed: int = 0,
    device: torch.device = torch.device("cpu"),
    use_prefetcher: bool = False,
):
    is_training_split = split == "train"

    if use_distributed_sampler:
        logger.info(f"Using DistributedSampler for {split} split")
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_training_split, seed=seed)
    elif is_training_split:
        generator = torch.Generator().manual_seed(seed)
        sampler = torch.utils.data.RandomSampler(dataset, replacement=False, generator=generator)
    else:
        logger.info(f"Using {split} dataloader with SequentialSampler")
        sampler = torch.utils.data.SequentialSampler(dataset)

    dataloader_lmbda = lambda x: DataLoader(
        dataset=dataset, batch_size=batch_size_per_gpu, num_workers=x, pin_memory=True, sampler=sampler
    )
    best_num_workers = find_best_num_workers(dataloader_lmbda)  # if num_workers == 0 else num_workers
    logger.info(f"Using `num_workers={best_num_workers}` for {split} dataloader")
    dataloader = dataloader_lmbda(best_num_workers)

    if use_prefetcher:
        if device.type == "cpu":
            raise ValueError("Using prefetcher with CPU runtime is not supported.")
        logger.info(f"Wrapping {split} dataloader in prefetcher")
        dataloader = PrefetchLoader(dataloader, device)

    logger.info(f"Dataloader for {split} split has {len(dataloader)} batches of size {batch_size_per_gpu}")

    return dataloader
