import re

import ml_collections as mlc

import shared
from configs.optim import OptimizerConfig
from src.constrained.cmp import (
    ERMClassificationProblem,
    ERMRegressionProblem,
    FeasibleClassificationProblem,
    FeasibleRegressionProblem,
    WeightedERMClassificationProblem,
)

MLC_PH = mlc.config_dict.config_dict.placeholder


def _basic_config():
    _config = mlc.ConfigDict()
    _config.cmp_class = MLC_PH(object)
    _config.cmp_kwargs = mlc.ConfigDict()
    return _config


def feasibility_task_config():
    _config = mlc.ConfigDict()

    _config.task = _basic_config()

    _config.task.pointwise_probability = MLC_PH(float)
    _config.task.pointwise_loss = MLC_PH(float)
    _config.task.early_stop_on_feasible = False

    _config.task.cmp_class = FeasibleClassificationProblem
    _config.task.cmp_kwargs = mlc.ConfigDict()
    _config.task.cmp_kwargs.use_strict_accuracy = False

    _config.task.multiplier_init_value = 0.0
    _config.task.multiplier_kwargs = mlc.ConfigDict()
    _config.task.multiplier_kwargs.restart_on_feasible = False
    _config.task.multiplier_kwargs.enforce_positive = True

    _config.optim = mlc.ConfigDict()
    _config.optim.cooper_optimizer_name = "AlternatingDualPrimalOptimizer"
    _config.optim.dual_optimizer = OptimizerConfig(name="SGD", shared_kwargs={"lr": 1e-2}).todict()

    return _config


def erm_task_config():
    _config = mlc.ConfigDict()

    _config.task = _basic_config()

    _config.task.pointwise_probability = 1.0
    _config.task.pointwise_loss = 0.0

    _config.task.cmp_class = ERMClassificationProblem
    _config.task.cmp_kwargs = mlc.ConfigDict()

    _config.optim = mlc.ConfigDict()
    _config.optim.cooper_optimizer_name = "UnconstrainedOptimizer"

    return _config


def weighted_erm_task_config():
    _config = mlc.ConfigDict()

    _config.task = _basic_config()

    _config.task.pointwise_probability = 1.0
    _config.task.cmp_class = WeightedERMClassificationProblem
    _config.task.cmp_kwargs = mlc.ConfigDict()
    _config.task.cmp_kwargs.class_weights = mlc.ConfigDict()
    _config.task.cmp_kwargs.class_weights.type = "effective"
    _config.task.cmp_kwargs.class_weights.beta = 0.9999
    _config.task.cmp_kwargs.class_weights.weights = None

    _config.optim = mlc.ConfigDict()
    _config.optim.cooper_optimizer_name = "UnconstrainedOptimizer"

    return _config


TASK_CONFIGS = {
    None: _basic_config,
    "feasibility": feasibility_task_config,
    "erm": erm_task_config,
    "weighted_erm": weighted_erm_task_config,
}


def get_config(config_string):
    """Examples for config_string:
    - "task=feasibility task.pointwise_probability=0.9 task.multiplier_kwargs.restart_on_feasible=False"
    - "task=erm pointwise_probability=0.9"
    """

    # Extract the key-value pairs from the config string which has the format
    # "key1=value1 key2=value2 ..."
    matches = re.findall(shared.REGEX_PATTERN, config_string)

    # Create a dictionary to store the extracted values
    variables = {key: value for key, value in matches}
    cluster = variables.pop("task")
    config_dict = TASK_CONFIGS[cluster]()

    shared.update_config_with_cli_args(config_dict, variables)

    return config_dict
