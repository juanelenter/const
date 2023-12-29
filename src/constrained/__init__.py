from typing import List

import cooper
import numpy as np
import torch
import wandb

import shared

from .cmp import (
    ERMClassificationProblem,
    ERMRegressionProblem,
    FeasibleClassificationProblem,
    FeasibleRegressionProblem,
)
from .cmp import ERMClassificationProblem, FeasibleClassificationProblem, WeightedERMClassificationProblem
from .optimizer_utils import build_cooper_optimizer_and_schedulers

logger = shared.fetch_main_logger()


def probability_to_cross_entropy_loss(pointwise_probability: float) -> float:
    """When using a cross-entropy loss, the following statements are equivalent:
    - imposing an upper-bound constraint of epsilon on the loss
    - imposing a lower-bound constraint of exp(-epsilon) on the probability of the *correct* class

    It is more intuitive/interpretable to think in terms of the requested probability of
    the model predicting the correct class, so we specify the probability as the task
    parameter and convert it to a loss here.

    Note that the CMPs have constraints on the loss.
    """
    if pointwise_probability is None:
        return 0.0
    else:
        if not (0 <= pointwise_probability <= 1):
            raise ValueError("Pointwise probability must be between 0 and 1")
        return np.log(1 / pointwise_probability)


def build_cmp(config, device=None, multiplier=None, num_training_samples=None, cmp_metadata=None) -> cooper.ConstrainedMinimizationProblem:

    if (config.task.cmp_class == ERMClassificationProblem) or (config.task.cmp_class == FeasibleClassificationProblem):
        pointwise_probability = config.task.pointwise_probability
        target_pointwise_loss = probability_to_cross_entropy_loss(pointwise_probability)
        logger.info(
            f"Provided pointwise probability {pointwise_probability} is equivalent to a target pointwise loss of {target_pointwise_loss}"
        )
    elif (config.task.cmp_class == ERMRegressionProblem) or (config.task.cmp_class == FeasibleRegressionProblem):
        target_pointwise_loss = config.task.pointwise_loss
        logger.info(f"Provided target pointwise loss of {target_pointwise_loss}")
    else: 
        raise ValueError(f"Unknown CMP class {config.task.cmp_class}")
    cmp_kwargs = {"target_pointwise_loss": target_pointwise_loss}
    cmp_kwargs.update(config.task.cmp_kwargs)

    if config.task.cmp_class.has_dual_variables:
        if multiplier is None:
            logger.info(
                f"Building IndexedMultiplier for {num_training_samples} samples with kwargs {config.task.multiplier_kwargs.to_dict()}"
            )
            init = config.task.multiplier_init_value + torch.zeros((num_training_samples, 1))
            multiplier = cooper.multipliers.IndexedMultiplier(init=init, **config.task.multiplier_kwargs)
        else:
            logger.info(f"Using provided multiplier.")

        if device is not None:
            multiplier = multiplier.to(device)
        cmp_kwargs["multiplier"] = multiplier

    if hasattr(config.task.cmp_class, "has_class_weights"):
        cmp_kwargs["dataset_metadata"] = cmp_metadata.dataset
        cmp_kwargs["device"] = device

    logger.info(f"Building {config.task.cmp_class.__name__} with kwargs: {cmp_kwargs}")
    cmp = config.task.cmp_class(**cmp_kwargs)

    if hasattr(config.task.cmp_class, "has_class_weights") and cmp_metadata.is_main_process:
        for c, weight in enumerate(cmp.class_weights):
            wandb.log({f"class_weight/{c}": weight.item()})

    return cmp
