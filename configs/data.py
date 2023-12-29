import os
import re

import ml_collections as mlc

import shared

MLC_PH = mlc.config_dict.config_dict.placeholder


def get_data_path():
    return os.environ.get("DATA_DIR", "~/datasets")


def _basic_config():
    _config = mlc.ConfigDict()

    _config.dataset_name = MLC_PH(str)
    _config.dataset_kwargs = mlc.ConfigDict()

    _config.label_noise = mlc.ConfigDict()
    _config.label_noise.enabled = False
    _config.label_noise.noise_ratio = MLC_PH(float)
    _config.label_noise.noise_map = MLC_PH(dict)
    _config.label_noise.seed = MLC_PH(int)

    _config.imbalance = mlc.ConfigDict()
    _config.imbalance.enabled = False
    _config.imbalance.class_ratios = MLC_PH(list)
    _config.imbalance.seed = MLC_PH(int)

    _config.dataloader = mlc.ConfigDict()
    # _config.dataloader.num_workers = MLC_PH(int)
    _config.dataloader.use_distributed_sampler = MLC_PH(bool)
    _config.dataloader.use_prefetcher = MLC_PH(bool)
    _config.dataloader.seed = MLC_PH(int)

    _config.batch_sizes_per_gpu = mlc.ConfigDict()
    _config.batch_sizes_per_gpu.train = MLC_PH(int)
    _config.batch_sizes_per_gpu.val = MLC_PH(int)

    return _config


def mnist_config():
    _config = _basic_config()

    _config.dataset_name = "mnist"

    _config.dataset_kwargs.data_path = get_data_path()

    _config.dataloader.use_distributed_sampler = False
    _config.dataloader.use_prefetcher = False
    _config.dataloader.seed = 0

    _config.batch_sizes_per_gpu.train = 128
    _config.batch_sizes_per_gpu.val = 128

    return _config


def mnist1d_config():
    _config = _basic_config()

    _config.dataset_name = "mnist1d"

    _config.dataloader.use_distributed_sampler = False
    _config.dataloader.use_prefetcher = False
    _config.dataloader.seed = 0

    _config.batch_sizes_per_gpu.train = 128
    _config.batch_sizes_per_gpu.val = 128

    _config.dataset_kwargs.data_path = get_data_path()
    # If `regenerate=False`, the other init_kwargs are ignored and the store/online dataset is used instead
    _config.dataset_kwargs.regenerate = False
    _config.dataset_kwargs.url = "https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl"

    # _config.dataset_kwargs.regenerate = True
    # _config.dataset_kwargs.num_samples = 12500
    # _config.dataset_kwargs.train_split = 0.8
    # _config.dataset_kwargs.template_len = 12
    # _config.dataset_kwargs.padding = [36, 60]
    # _config.dataset_kwargs.scale_coeff = 0.4
    # _config.dataset_kwargs.max_translation = 48
    # _config.dataset_kwargs.corr_noise_scale = 0.25
    # _config.dataset_kwargs.iid_noise_scale = 2e-2
    # _config.dataset_kwargs.shear_scale = 0.75
    # _config.dataset_kwargs.shuffle_seq = False
    # _config.dataset_kwargs.final_seq_length = 40

    return _config


def cifar10_config():
    _config = _basic_config()

    _config.dataset_name = "cifar10"
    _config.dataset_kwargs.data_path = get_data_path()
    _config.dataset_kwargs.use_data_augmentation = True

    _config.dataloader.use_distributed_sampler = False
    _config.dataloader_seed = 0
    _config.dataloader.use_prefetcher = False
    _config.init_kwargs = mlc.ConfigDict()
    _config.init_kwargs.noise_level = 0.0

    _config.imbalance = mlc.ConfigDict()
    _config.imbalance.kwargs = mlc.ConfigDict()
    _config.imbalance.enabled = False
    _config.imbalance.class_ratios = MLC_PH(list)
    _config.imbalance.seed = MLC_PH(int)

    _config.batch_sizes_per_gpu = mlc.ConfigDict()
    _config.batch_sizes_per_gpu.train = 128
    _config.batch_sizes_per_gpu.val = 128

    _config.dataloader.use_distributed_sampler = False
    _config.dataloader.use_prefetcher = False
    _config.dataloader.seed = 0

    return _config


def cifar10_lt_config():
    _config = _basic_config()

    _config.dataset_name = "cifar10"
    _config.dataset_kwargs.data_path = get_data_path()
    _config.dataloader.use_distributed_sampler = False
    _config.dataloader_seed = 0
    _config.dataloader.use_prefetcher = False
    _config.init_kwargs = mlc.ConfigDict()
    _config.init_kwargs.noise_level = 0.0

    _config.imbalance.kwargs = mlc.ConfigDict()
    _config.imbalance.kwargs.type = "exp"
    _config.imbalance.kwargs.ratio = 0.1
    _config.imbalance.enabled = True

    _config.dataloader.use_distributed_sampler = False
    _config.dataloader.use_prefetcher = False
    _config.dataloader.seed = 0

    _config.batch_sizes_per_gpu = mlc.ConfigDict()
    _config.batch_sizes_per_gpu.train = 128
    _config.batch_sizes_per_gpu.val = 128

    return _config


def imagenet_config():
    _config = _basic_config()

    _config.dataset_name = "imagenet"
    _config.dataset_kwargs.data_path = None
    _config.dataset_kwargs.use_data_augmentation = True

    _config.dataloader.use_distributed_sampler = True
    _config.dataloader.use_prefetcher = True
    _config.dataloader.seed = 0

    _config.batch_sizes_per_gpu.train = 128
    _config.batch_sizes_per_gpu.val = 128

    return _config


def two_moons_config():
    _config = _basic_config()

    _config.dataset_name = "two_moons"

    _config.dataset_kwargs.train_samples = 10000
    _config.dataset_kwargs.train_noise = 0.1

    _config.dataset_kwargs.val_samples = 5000
    _config.dataset_kwargs.val_noise = 0.2

    _config.dataset_kwargs.imbalance_ratio = 0.0

    _config.dataloader.use_distributed_sampler = False
    _config.dataloader.use_prefetcher = False
    _config.dataloader.seed = 0

    _config.batch_sizes_per_gpu.train = 128
    _config.batch_sizes_per_gpu.val = 128

    return _config


def two_circles_config():
    _config = _basic_config()

    _config.dataset_name = "two_circles"

    _config.dataset_kwargs.factor = 0.8

    _config.dataset_kwargs.n_circles = 4

    _config.dataset_kwargs.train_samples = 5000
    _config.dataset_kwargs.train_noise = 0.05

    _config.dataset_kwargs.val_samples = 5000
    _config.dataset_kwargs.val_noise = 0.05

    _config.dataset_kwargs.imbalance_ratio = 0.1

    _config.dataloader.use_distributed_sampler = False
    _config.dataloader.use_prefetcher = False
    _config.dataloader.seed = 0

    _config.batch_sizes_per_gpu.train = 128
    _config.batch_sizes_per_gpu.val = 128

    return _config


def linsep_2d():
    _config = _basic_config()

    _config.dataset_name = "linearly_separable"

    _config.dataset_kwargs.train_samples = 500
    _config.dataset_kwargs.val_samples = 1000

    _config.dataset_kwargs.dim = 2
    _config.dataset_kwargs.margin = 1.0

    _config.dataset_kwargs.data_seed = 0
    _config.dataset_kwargs.sample_seed = 0

    _config.dataset_kwargs.positive_proportion = 0.5

    _config.dataloader.use_distributed_sampler = False
    _config.dataloader.use_prefetcher = False
    _config.dataloader.seed = 0

    _config.batch_sizes_per_gpu.train = 128
    _config.batch_sizes_per_gpu.val = 128

    return _config


def speechcommands_config():
    _config = _basic_config()

    _config.dataset_name = "speechcommands"

    _config.dataset_kwargs.data_path = get_data_path()

    _config.dataloader.use_distributed_sampler = False
    _config.dataloader.use_prefetcher = False
    _config.dataloader.seed = 0

    _config.batch_sizes_per_gpu.train = 128
    _config.batch_sizes_per_gpu.val = 128

    return _config


def poverty_config():
    _config = _basic_config()

    _config.dataset_name = "poverty"
    _config.dataset_kwargs.data_path = get_data_path()

    _config.dataloader.use_distributed_sampler = False
    _config.dataloader.use_prefetcher = False
    _config.dataloader.seed = 0

    _config.batch_sizes_per_gpu.train = 64
    _config.batch_sizes_per_gpu.val = 64

    return _config


def calihousing_config():
    _config = _basic_config()

    _config.dataset_name = "calihousing"
    _config.dataset_kwargs.data_path = get_data_path()

    _config.dataloader.use_distributed_sampler = False
    _config.dataloader.use_prefetcher = False
    _config.dataloader.seed = 0

    _config.batch_sizes_per_gpu.train = 128
    _config.batch_sizes_per_gpu.val = 128

    return _config


DATA_CONFIGS = {
    None: _basic_config,
    "mnist1d": mnist1d_config,
    "mnist": mnist_config,
    "imagenet": imagenet_config,
    "two_moons": two_moons_config,
    "two_circles": two_circles_config,
    "linsep_2d": linsep_2d,
    "speechcommands": speechcommands_config,
    "cifar10": cifar10_config,
    "cifar10_lt": cifar10_lt_config,
    "poverty": poverty_config,
    "calihousing": calihousing_config,
}


def get_config(config_string=None):
    # Extract the key-value pairs from the config string which has the format
    # "key1=value1 key2=value2 ..."
    matches = re.findall(shared.REGEX_PATTERN, config_string)

    # Create a dictionary to store the extracted values
    variables = {key: value for key, value in matches}
    data_name = variables.pop("data")
    config_dict = {"data": DATA_CONFIGS[data_name]()}

    shared.update_config_with_cli_args(config_dict, variables)

    return config_dict
