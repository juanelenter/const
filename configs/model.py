import re

import ml_collections as mlc
import torch

import shared
from configs.optim import OptimizerConfig

MLC_PH = mlc.config_dict.config_dict.placeholder


def _basic_config():
    _config = mlc.ConfigDict()
    _config.name = MLC_PH(str)
    _config.init_seed = MLC_PH(int)
    _config.init_kwargs = mlc.ConfigDict()

    return _config


def TwoDim_MLP_config():
    _config = mlc.ConfigDict()

    _config.model = _basic_config()
    _config.model.name = "MLP"
    _config.model.init_seed = 0

    _config.model.init_kwargs.activation_type = torch.nn.ReLU
    _config.model.init_kwargs.input_size = 2
    _config.model.init_kwargs.output_size = 2
    _config.model.init_kwargs.hidden_sizes = MLC_PH(tuple)

    primal_optimizer_config = OptimizerConfig(name="AdamW", shared_kwargs={"lr": 1e-3}).todict()
    _config.optim = mlc.ConfigDict({"primal_optimizer": primal_optimizer_config})

    return _config


def MNIST1D_MLP():
    _config = mlc.ConfigDict()

    _config.model = _basic_config()
    _config.model.name = "MLP"
    _config.model.init_seed = 0

    _config.model.init_kwargs.activation_type = torch.nn.ReLU
    _config.model.init_kwargs.input_size = 40
    _config.model.init_kwargs.output_size = 10
    _config.model.init_kwargs.hidden_sizes = MLC_PH(tuple)

    primal_optimizer_config = OptimizerConfig(name="Adam", shared_kwargs={"lr": 1e-2}).todict()
    _config.optim = mlc.ConfigDict({"primal_optimizer": primal_optimizer_config})

    return _config


def MNIST1D_CNN():
    _config = mlc.ConfigDict()

    _config.model = _basic_config()
    _config.model.name = "MNIST1DCNN"
    _config.model.init_seed = 0

    _config.model.init_kwargs.output_size = 10

    primal_optimizer_config = OptimizerConfig(name="AdamW", shared_kwargs={"lr": 1e-3}).todict()
    _config.optim = mlc.ConfigDict({"primal_optimizer": primal_optimizer_config})

    return _config


def TwoDim_LogisticRegression_config():
    _config = mlc.ConfigDict()

    _config.model = _basic_config()
    _config.model.name = "MLP"
    _config.model.init_seed = 0

    _config.model.init_kwargs.activation_type = torch.nn.Identity
    _config.model.init_kwargs.input_size = 2
    _config.model.init_kwargs.output_size = 2
    _config.model.init_kwargs.hidden_sizes = []

    primal_optimizer_config = OptimizerConfig(name="SGD", shared_kwargs={"lr": 1e-3}).todict()
    _config.optim = mlc.ConfigDict({"primal_optimizer": primal_optimizer_config})

    return _config


def MNIST_CNN_config():
    _config = mlc.ConfigDict()

    _config.model = _basic_config()
    _config.model.name = "MNISTCNN"
    _config.model.init_seed = 0

    primal_optimizer_config = OptimizerConfig(name="AdamW", shared_kwargs={"lr": 5e-4}).todict()
    _config.optim = mlc.ConfigDict({"primal_optimizer": primal_optimizer_config})

    return _config


def MNIST_ResNet18_config():
    _config = mlc.ConfigDict()

    _config.model = _basic_config()
    _config.model.init_seed = 0

    _config.model.name = "ResNet18MNIST"
    _config.model.init_kwargs.weights = None

    primal_optimizer_config = OptimizerConfig(name="AdamW", shared_kwargs={"lr": 1e-3}).todict()
    _config.optim = mlc.ConfigDict({"primal_optimizer": primal_optimizer_config})

    return _config


def ImageNet_ResNet50_config():
    _config = mlc.ConfigDict()

    _config.model = _basic_config()
    _config.model.init_seed = 0

    _config.model.name = "ResNet"
    _config.model.init_kwargs.num_classes = 1000
    _config.model.init_kwargs.resnet_name = "resnet50"
    _config.model.init_kwargs.weights = None

    primal_optimizer_config = OptimizerConfig(
        name="SGD", shared_kwargs={"lr": 1e-3, "momentum": 0.9, "nesterov": True}
    ).todict()
    _config.optim = mlc.ConfigDict({"primal_optimizer": primal_optimizer_config})

    return _config


def AudioCNN_config():
    _config = mlc.ConfigDict()

    _config.model = _basic_config()
    _config.model.name = "AudioCNN"
    _config.model.init_seed = 0
    _config.model.init_kwargs.n_channel = 32
    primal_optimizer_config = OptimizerConfig(name="Adam", shared_kwargs={"lr": 1e-3, "weight_decay": 1e-4}).todict()
    _config.optim = mlc.ConfigDict({"primal_optimizer": primal_optimizer_config})

    return _config


def CIFAR_ResNet18_config():
    _config = mlc.ConfigDict()

    _config.model = _basic_config()
    _config.model.init_seed = 0

    _config.model.name = "ResNet18CIFAR"
    _config.model.init_kwargs.num_classes = 10
    _config.model.init_kwargs.weights = None

    primal_optimizer_config = OptimizerConfig(name="AdamW", shared_kwargs={"lr": 1e-3}).todict()
    _config.optim = mlc.ConfigDict({"primal_optimizer": primal_optimizer_config})

    return _config


def Poverty_ResNet18_config():
    _config = mlc.ConfigDict()

    _config.model = _basic_config()
    _config.model.name = "ResNet18MS"
    _config.model.init_seed = 0

    _config.model.init_kwargs.num_channels = 8
    _config.model.init_kwargs.num_classes = 1

    primal_optimizer_config = OptimizerConfig(name="Adam", shared_kwargs={"lr": 1e-3, "weight_decay": 0}).todict()

    _config.optim = mlc.ConfigDict({"primal_optimizer": primal_optimizer_config})

    return _config


def CaliHousing_MLP_config():
    _config = mlc.ConfigDict()

    _config.model = _basic_config()
    _config.model.name = "MLP"
    _config.model.init_seed = 0

    _config.model.init_kwargs.activation_type = torch.nn.ReLU
    _config.model.init_kwargs.input_size = 8
    _config.model.init_kwargs.output_size = 1
    _config.model.init_kwargs.hidden_sizes = MLC_PH(tuple)

    primal_optimizer_config = OptimizerConfig(name="Adam", shared_kwargs={"lr": 1e-3}).todict()
    _config.optim = mlc.ConfigDict({"primal_optimizer": primal_optimizer_config})

    return _config


MODEL_CONFIGS = {
    "TwoDim_MLP": TwoDim_MLP_config,
    "TwoDim_LogisticRegression": TwoDim_LogisticRegression_config,
    "MNIST_CNN": MNIST_CNN_config,
    "MNIST_ResNet18": MNIST_ResNet18_config,
    "MNIST1D_MLP": MNIST1D_MLP,
    "MNIST1D_CNN": MNIST1D_CNN,
    "CIFAR_ResNet18": CIFAR_ResNet18_config,
    "ImageNet_ResNet50": ImageNet_ResNet50_config,
    "AudioCNN": AudioCNN_config,
    "Poverty_ResNet18": Poverty_ResNet18_config,
    "CaliHousing_MLP": CaliHousing_MLP_config,
}


def get_config(config_string):
    """Examples for config_string:
    - "model=TwoDim_MLP model.init_kwargs.hidden_sizes=() optim.primal_optimizer.shared_kwargs.lr=1e-1"
    - "model=MNIST_CNN optim.primal_optimizer.shared_kwargs.lr=1e-3"
    """
    # Extract the key-value pairs from the config string which has the format
    # "key1=value1 key2=value2 ..."
    matches = re.findall(shared.REGEX_PATTERN, config_string)

    # Create a dictionary to store the extracted values
    variables = {key: value for key, value in matches}
    model_name = variables.pop("model")
    config_dict = MODEL_CONFIGS[model_name]()

    shared.update_config_with_cli_args(config_dict, variables)

    return config_dict
