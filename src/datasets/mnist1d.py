import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import shared

logger = shared.fetch_main_logger()


class MNIST1D:
    name = "MNIST1D"

    def __init__(self, data_path, config):
        logger.info("Initializing MNIST1D dataset")

        shuffle_str = "_shuffle" if config is not None and getattr(config, "shuffle_seq", False) else ""
        path = Path(data_path) / "MNIST1D/mnist1d_data{}.pkl".format(shuffle_str)
        path = path.expanduser().resolve()

        if config.regenerate:
            logger.info("Did or could not load data from {}. Rebuilding dataset...".format(path))
            dataset_dict = make_dataset(config)
            to_pickle(dataset_dict, path)
        else:
            if os.path.exists(path):
                logger.info("File already exists. Skipping download.")
            else:
                logger.info("Downloading MNIST1D dataset from {}".format(config.url))
                r = requests.get(config.url, allow_redirects=True)
                open(path, "wb").write(r.content)
                logger.info("Saving to {}".format(path))
            dataset_dict = from_pickle(path)
            logger.info("Successfully loaded data from {}".format(path))

        self.dataset_dict = dataset_dict


def load_mnist1d_dataset(data_path, config):
    dataset = MNIST1D(data_path, config)

    dataset_namespace = SimpleNamespace()

    for split, x_key, y_key in [("train", "x", "y"), ("val", "x_val", "y_val")]:
        X = torch.tensor(dataset.dataset_dict[x_key], dtype=torch.float32)
        y = torch.tensor(dataset.dataset_dict[y_key], dtype=torch.long)
        split_dataset = torch.utils.data.TensorDataset(X, y)
        setattr(dataset_namespace, split, split_dataset)

    return dataset_namespace, dataset.num_classes


# --------------------------------------------------------------------------------------
# Functions below taken from https://github.com/greydanus/mnist1d
# --------------------------------------------------------------------------------------

import requests
import torch
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter


def pad(x, padding):
    low, high = padding
    p = low + int(np.random.rand() * (high - low + 1))
    return np.concatenate([x, np.zeros((p))])


def shear(x, scale=10):
    coeff = scale * (np.random.rand() - 0.5)
    return x - coeff * np.linspace(-0.5, 0.5, len(x))


def translate(x, max_translation):
    k = np.random.choice(max_translation)
    return np.concatenate([x[-k:], x[:-k]])


def corr_noise_like(x, scale):
    noise = scale * np.random.randn(*x.shape)
    return gaussian_filter(noise, 2)


def iid_noise_like(x, scale):
    noise = scale * np.random.randn(*x.shape)
    return noise


def interpolate(x, N):
    scale = np.linspace(0, 1, len(x))
    new_scale = np.linspace(0, 1, N)
    new_x = interp1d(scale, x, axis=0, kind="linear")(new_scale)
    return new_x


def transform(x, y, config, eps=1e-8):
    new_x = pad(x + eps, config.padding)  # pad
    new_x = interpolate(new_x, config.template_len + config.padding[-1])  # dilate
    new_y = interpolate(y, config.template_len + config.padding[-1])
    new_x *= 1 + config.scale_coeff * (np.random.rand() - 0.5)  # scale
    new_x = translate(new_x, config.max_translation)  # translate

    # add noise
    mask = new_x != 0
    new_x = mask * new_x + (1 - mask) * corr_noise_like(new_x, config.corr_noise_scale)
    new_x = new_x + iid_noise_like(new_x, config.iid_noise_scale)

    # shear and interpolate
    new_x = shear(new_x, config.shear_scale)
    new_x = interpolate(new_x, config.final_seq_length)  # subsample
    new_y = interpolate(new_y, config.final_seq_length)
    return new_x, new_y


def to_pickle(thing, path):  # save something
    with open(path, "wb") as handle:
        pickle.dump(thing, handle, protocol=3)


def from_pickle(path):  # load something
    thing = None
    with open(path, "rb") as handle:
        thing = pickle.load(handle)
    return thing


# basic 1D templates for the 10 digits
def get_templates():
    d0 = np.asarray([5, 6, 6.5, 6.75, 7, 7, 7, 7, 6.75, 6.5, 6, 5])
    d1 = np.asarray([5, 3, 3, 3.4, 3.8, 4.2, 4.6, 5, 5.4, 5.8, 5, 5])
    d2 = np.asarray([5, 6, 6.5, 6.5, 6, 5.25, 4.75, 4, 3.5, 3.5, 4, 5])
    d3 = np.asarray([5, 6, 6.5, 6.5, 6, 5, 5, 6, 6.5, 6.5, 6, 5])
    d4 = np.asarray([5, 4.4, 3.8, 3.2, 2.6, 2.6, 5, 5, 5, 5, 5, 5])
    d5 = np.asarray([5, 3, 3, 3, 3, 5, 6, 6.5, 6.5, 6, 4.5, 5])
    d6 = np.asarray([5, 4, 3.5, 3.25, 3, 3, 3, 3, 3.25, 3.5, 4, 5])
    d7 = np.asarray([5, 7, 7, 6.6, 6.2, 5.8, 5.4, 5, 4.6, 4.2, 5, 5])
    d8 = np.asarray([5, 4, 3.5, 3.5, 4, 5, 5, 4, 3.5, 3.5, 4, 5])
    d9 = np.asarray([5, 4, 3.5, 3.5, 4, 5, 5, 5, 5, 4.7, 4.3, 5])

    x = np.stack([d0, d1, d2, d3, d4, d5, d6, d7, d8, d9])
    x -= x.mean(1, keepdims=True)  # whiten
    x /= x.std(1, keepdims=True)
    x -= x[:, :1]  # signal starts and ends at 0

    templates = {"x": x / 6.0, "t": np.linspace(-5, 5, len(d0)) / 6.0, "y": np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}
    return templates


# make a dataset
def make_dataset(config):
    templates = get_templates()

    xs, ys = [], []
    samples_per_class = config.num_samples // len(templates["y"])
    for label_ix in range(len(templates["y"])):
        for example_ix in range(samples_per_class):
            x = templates["x"][label_ix]
            t = templates["t"]
            y = templates["y"][label_ix]
            x, new_t = transform(x, t, config)  # new_t transformation is same each time
            xs.append(x)
            ys.append(y)

    batch_shuffle = np.random.permutation(len(ys))  # shuffle batch dimension
    xs = np.stack(xs)[batch_shuffle]
    ys = np.stack(ys)[batch_shuffle]

    if config.shuffle_seq:  # maybe shuffle the spatial dimension
        seq_shuffle = np.random.permutation(config.final_seq_length)
        xs = xs[..., seq_shuffle]

    new_t = new_t / xs.std()
    xs = (xs - xs.mean()) / xs.std()  # center the dataset & set standard deviation to 1

    # train / val split
    split_ix = int(len(ys) * config.train_split)
    dataset = {
        "x": xs[:split_ix],
        "x_val": xs[split_ix:],
        "y": ys[:split_ix],
        "y_val": ys[split_ix:],
        "t": new_t,
        "templates": templates,
    }
    return dataset
