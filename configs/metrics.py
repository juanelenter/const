import copy
from functools import partial
from types import SimpleNamespace

import ml_collections as mlc

from src.utils.meters import AverageMeter, BestMeter, StatsMeter

MLC_PH = mlc.config_dict.config_dict.placeholder

DEFAULT_AVG_KWARGS = {"meter_class": AverageMeter}
DEFAULT_MAX_KWARGS = {"meter_class": partial(BestMeter, direction="max")}
DEFAULT_STATS_KWARGS = {"meter_class": partial(StatsMeter, stats=["pos_avg", "nonpos_rate", "max"])}
AVG_STATS_KWARGS = {"meter_class": partial(StatsMeter, stats=["avg"])}


def _basic_config():
    metrics_config = mlc.ConfigDict()
    metrics_config.batch_level = MLC_PH(list)
    metrics_config.epoch_level = mlc.ConfigDict()
    return metrics_config


def classification_config():
    metrics_config = _basic_config()
    metrics_config.batch_level = ["avg_acc", "avg_loss", "max_loss"]

    basic_classifcation_metrics = [
        SimpleNamespace(name="Accuracy", log_name="avg_acc", kwargs=DEFAULT_AVG_KWARGS),
        SimpleNamespace(name="PerClassTop1Accuracy", log_name="class_acc", kwargs=DEFAULT_AVG_KWARGS),
        SimpleNamespace(name="CrossEntropy", log_name="loss", kwargs=AVG_STATS_KWARGS),
        SimpleNamespace(name="Violation", log_name="violation", kwargs=DEFAULT_STATS_KWARGS),
    ]

    metrics_config.epoch_level.train = copy.deepcopy(basic_classifcation_metrics)
    metrics_config.epoch_level.val = copy.deepcopy(basic_classifcation_metrics)

    return metrics_config


def imbalanced_classification_config():
    precision_namespace = SimpleNamespace(name="PerClassPrecision", log_name="class_prec", kwargs=DEFAULT_AVG_KWARGS)
    metrics_config = classification_config()
    metrics_config.epoch_level.train.append(precision_namespace)
    metrics_config.epoch_level.val.append(precision_namespace)

    return metrics_config


def large_scale_classification_config():
    """This is a configuration for large-scale classification tasks, where the training
    set is too large to compute the statistics for all training samples at every epoch,
    as in ImageNet.
    """
    metrics_config = classification_config()
    # Clear the pre-existing epoch-level metrics in training
    metrics_config.epoch_level.train = []
    return metrics_config


def regression_config():
    metrics_config = _basic_config()
    metrics_config.batch_level = ["avg_loss", "max_loss"]

    metrics_config.epoch_level.train = [
        SimpleNamespace(name="L2Loss", log_name="avg_loss", kwargs=DEFAULT_AVG_KWARGS),
        SimpleNamespace(name="Violation", log_name="violation", kwargs=DEFAULT_STATS_KWARGS),
    ]
    metrics_config.epoch_level.val = [
        SimpleNamespace(name="L2Loss", log_name="avg_loss", kwargs=DEFAULT_AVG_KWARGS),
        SimpleNamespace(name="Violation", log_name="violation", kwargs=DEFAULT_STATS_KWARGS),
    ]
    return metrics_config


METRICS_CONFIGS = {
    "classification": classification_config,
    "imbalanced_classification": imbalanced_classification_config,
    "large_scale_classification": large_scale_classification_config,
    "regression": regression_config,
    None: _basic_config,
}


def get_config(config_string=None):
    return {"metrics": METRICS_CONFIGS[config_string]()}
