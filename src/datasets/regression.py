from types import SimpleNamespace

import numpy as np
import sklearn.datasets
import torch
import torchvision.transforms as transforms
import wilds
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from wilds.datasets.wilds_dataset import WILDSDataset

import shared

logger = shared.fetch_main_logger()


def load_calihousing_dataset(data_path):
    logger.info(f"Loading California Housing dataset from {data_path}")

    X, y = fetch_california_housing(data_home=data_path, return_X_y=True, download_if_missing=True)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()

    #(TODO:juanelenter) Compute this once and copy the values.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_mean = torch.mean(X_train, axis=0)
    X_std = torch.std(X_train, axis=0)
    y_mean = torch.mean(y_train, axis=0)
    y_std = torch.std(y_train, axis=0)
    X_train = (X_train - X_mean) / X_std
    y_train = (y_train - y_mean) / y_std
    X_test = (X_test - X_mean) / X_std
    y_test = (y_test - y_mean) / y_std

    train_dataset = MyCaliHousing(X_train, y_train)
    val_dataset = MyCaliHousing(X_test, y_test)

    return SimpleNamespace(train=train_dataset, val=val_dataset)


class MyCaliHousing(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y.unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_poverty_dataset(data_path):
    logger.info(f"Loading WILDS- Poverty dataset from {data_path}")

    dataset = wilds.get_dataset("poverty", download=True, root_dir=data_path)
    dataset.get_subset = get_subset
    train_transform = transforms.Compose([])  # Their paper says normalize it, but their code doesn't.
    val_transform = transforms.Compose([])
    train_dataset = dataset.get_subset(dataset, frac=1, split="train", transform=train_transform)
    val_dataset = dataset.get_subset(
        dataset, frac=1, split="val", transform=val_transform
    )  # there's test too, could merge val and train and use test as val.
    return SimpleNamespace(train=train_dataset, val=val_dataset)


class MyWILDSSubset(WILDSDataset):
    def __init__(self, dataset, indices, transform, do_transform_y=False):
        """
        This acts like `torch.utils.data.Subset`, but on `WILDSDatasets`.
        We pass in `transform` (which is used for data augmentation) explicitly
        because it can potentially vary on the training vs. test subsets.

        `do_transform_y` (bool): When this is false (the default),
                                 `self.transform ` acts only on  `x`.
                                 Set this to true if `self.transform` should
                                 operate on `(x,y)` instead of just `x`.
        """
        self.dataset = dataset
        self.indices = indices
        inherited_attrs = [
            "_dataset_name",
            "_data_dir",
            "_collate",
            "_split_scheme",
            "_split_dict",
            "_split_names",
            "_y_size",
            "_n_classes",
            "_metadata_fields",
            "_metadata_map",
        ]
        for attr_name in inherited_attrs:
            if hasattr(dataset, attr_name):
                setattr(self, attr_name, getattr(dataset, attr_name))
        self.transform = transform
        self.do_transform_y = do_transform_y
        self.mean_y = 0.0551
        self.std_y = 0.8081

    def __getitem__(self, idx):
        x, y, metadata = self.dataset[self.indices[idx]]
        if self.transform is not None:
            if self.do_transform_y:
                x, y = self.transform(x, y)
            else:
                x = self.transform(x)
        y = (y - self.mean_y) / self.std_y
        return x, y

    def __len__(self):
        return len(self.indices)

    @property
    def split_array(self):
        return self.dataset._split_array[self.indices]

    @property
    def y_array(self):
        return self.dataset._y_array[self.indices]

    @property
    def metadata_array(self):
        return self.dataset.metadata_array[self.indices]

    def eval(self, y_pred, y_true, metadata):
        return self.dataset.eval(y_pred, y_true, metadata)


def get_subset(self, split, frac=1.0, transform=None):
    """
    Args:
        - split (str): Split identifier, e.g., 'train', 'val', 'test'.
                        Must be in self.split_dict.
        - frac (float): What fraction of the split to randomly sample.
                        Used for fast development on a small dataset.
        - transform (function): Any data transformations to be applied to the input x.
    Output:
        - subset (WILDSSubset): A (potentially subsampled) subset of the WILDSDataset.
    """
    if split not in self.split_dict:
        raise ValueError(f"Split {split} not found in dataset's split_dict.")

    split_mask = self.split_array == self.split_dict[split]
    split_idx = np.where(split_mask)[0]

    if frac < 1.0:
        # Randomly sample a fraction of the split
        num_to_retain = int(np.round(float(len(split_idx)) * frac))
        split_idx = np.sort(np.random.permutation(split_idx)[:num_to_retain])

    return MyWILDSSubset(self, split_idx, transform)
