import os
from types import SimpleNamespace

import torch

import shared
from src.utils import scan_namespace
from src.utils.distributed import wait_for_all_processes

from .audio import load_speechcommands_dataset
from .linearly_separable import LinearlySeparableDataset, load_linearly_separable_dataset
from .mnist1d import MNIST1D, load_mnist1d_dataset
from .prefetch_loader import PrefetchLoader
from .regression import load_calihousing_dataset, load_poverty_dataset
from .two_circles import TwoCirclesDataset, load_two_circles_dataset
from .two_moons import TwoMoonsDataset, load_two_moons_dataset
from .utils import (
    IndexedDataset,
    apply_classification_label_imbalance,
    apply_classification_label_noise_,
    create_dataloader_for_dataset,
)
from .vision import (
    copy_imagenet_debug_to_tmpdir,
    copy_imagenet_to_tmpdir,
    load_cifar10_dataset,
    load_imagenet_dataset,
    load_mnist_dataset,
)

# Define a list of two-dimensional datasets
TWO_DIM_DATASETS = [TwoCirclesDataset, TwoMoonsDataset, LinearlySeparableDataset]

# Define a dictionary of known dataset loaders
KNOWN_DATASET_LOADERS = {
    "mnist": load_mnist_dataset,
    "mnist1d": load_mnist1d_dataset,
    "cifar10": load_cifar10_dataset,
    "imagenet": load_imagenet_dataset,
    "two_circles": load_two_circles_dataset,
    "two_moons": load_two_moons_dataset,
    "linearly_separable": load_linearly_separable_dataset,
    "speechcommands": load_speechcommands_dataset,
    "poverty": load_poverty_dataset,
    "calihousing": load_calihousing_dataset,
}

logger = shared.fetch_main_logger()


def build_datasets(config, is_main_process) -> tuple[SimpleNamespace, SimpleNamespace]:
    """
    Builds and returns datasets and metadata for a given configuration.

    Notes:
        - The final datasets are wrapped in IndexedDataset.
        - The metadata is a SimpleNamespace object containing the following attributes:
            - corrupted_train_idxs: A list of indices of the corrupted training samples.
        - Transformation like label noise and imbalance are applied at the dataset level
            and are performed in a deterministic manner based on prescribed seeds.
        - These transformations are currently only supported for classification tasks.
        - Dataset imbalance is applied _after_ label noise since it looks at the sample
            labels to sub-sample the dataset.
    """
    dataset_name = config.data.dataset_name

    if dataset_name == "imagenet" and is_main_process:
        copy_imagenet_debug_to_tmpdir(os.getcwd())

        # TODO(gallego-posada): Make sure full ImageNet copying is happening correctly
        # copy_imagenet_to_tmpdir(os.getcwd())
    wait_for_all_processes()

    dataset_namespace, num_classes = KNOWN_DATASET_LOADERS[dataset_name](**config.data.dataset_kwargs)
    dataset_metadata = SimpleNamespace(num_classes=num_classes)

    if config.data.label_noise.enabled or config.data.imbalance.enabled:
        if "train" not in config.data.batch_sizes_per_gpu:
            raise ValueError("Requested data poisoining or imbalance but split `train` has no specified batchsize.")

    if config.data.label_noise.enabled:
        # We only corrupt samples in the training set. This transformation happens in-place.
        dataset_metadata.corrupted_train_idxs = apply_classification_label_noise_(
            dataset=dataset_namespace.train, **config.data.label_noise
        )

    if config.data.imbalance.enabled:
        dataset_namespace.train, dataset_metadata.imbalance_metadata = apply_classification_label_imbalance(
            dataset=dataset_namespace.train, **config.data.imbalance.kwargs
        )

    # Wrap all datasets in an IndexedDataset, which allows us keep track of the sample
    # indices for the multipliers.
    indexed_dataset_namespace = SimpleNamespace()
    for split, dataset in scan_namespace(dataset_namespace):
        setattr(indexed_dataset_namespace, split, IndexedDataset(dataset))

    return indexed_dataset_namespace, dataset_metadata


def build_dataloaders(dataset_namespace, config, device: torch.device, dist: SimpleNamespace) -> SimpleNamespace:
    """
    Builds and returns dataloaders for a given dataset_namespace and config.

    Returns:
        A SimpleNamespace object containing dataloaders.
    """
    if config.data.dataloader.use_distributed_sampler and not dist.multi_gpu:
        raise ValueError("Distributed sampler requires multi-gpu training.")

    shared_dataloader_kwargs = dict(device=device)
    shared_dataloader_kwargs.update(config.data.dataloader.to_dict())

    dataloaders = {}
    for split, batch_size_per_gpu in config.data.batch_sizes_per_gpu.items():
        split_dataset = getattr(dataset_namespace, split)
        logger.info(f"{config.data.dataset_name} dataset {split} split contains {len(split_dataset)} samples")

        dataloader = create_dataloader_for_dataset(
            dataset=split_dataset,
            split=split,
            batch_size_per_gpu=batch_size_per_gpu,
            **shared_dataloader_kwargs,
        )
        dataloaders[split] = dataloader

        log_message = f"Initialized {split} dataloader of length {len(dataloader)} with batch size {batch_size_per_gpu}"
        if hasattr(dataloader, "num_workers"):
            log_message += f" and {dataloader.num_workers} workers"
        logger.info(log_message)

    return SimpleNamespace(**dataloaders)
