import cooper
import torch

import shared

logger = shared.fetch_main_logger()


def build_optimizer_from_config(model: torch.nn.Module, optimizer_config, parameter_groups=None):
    # TODO(juan43ramirez):could split this function into two, one that takes a model and another that takes
    # parameter_groups. This would help with clarity.

    if model is not None and parameter_groups is not None:
        raise ValueError("Exactly one of `model` and `parameter_groups` must be provided")

    if parameter_groups is None:
        parameter_groups = model.parameter_groups()

    try:
        optimizer_class = torch.optim.__dict__[optimizer_config.name]
    except KeyError:
        optimizer_class = cooper.optim.__dict__[optimizer_config.name]

    is_per_group_kwargs_empty = optimizer_config.per_group_kwargs is None or len(optimizer_config.per_group_kwargs) == 0
    if is_per_group_kwargs_empty and model is not None:
        grouped_params_with_kwargs = [{"params": model.parameters()}]
        group_names = ["params"]
    else:
        if is_per_group_kwargs_empty:
            group_names = list(parameter_groups.keys())
            per_group_kwargs = {key: {} for key in group_names}
        else:
            per_group_kwargs = optimizer_config.per_group_kwargs
            group_names = list(optimizer_config.per_group_kwargs.keys())

        # We are working based on parameter_groups
        grouped_params_with_kwargs = []
        for group_name in group_names:
            grouped_params_with_kwargs.append({"params": parameter_groups[group_name], **per_group_kwargs[group_name]})

    optimizer = optimizer_class(grouped_params_with_kwargs, **optimizer_config.shared_kwargs)

    total_params_in_optimizer = 0
    logger.info(f"Created {optimizer_config.name} optimizer for the following groups:")
    for group_name, param_group in zip(group_names, optimizer.param_groups):
        num_params_in_group = sum([param.numel() for param in param_group["params"]])
        logger.info(f"  - {group_name}: {num_params_in_group} parameters")
        total_params_in_optimizer += num_params_in_group

    if optimizer_config.scheduler.name is not None:
        scheduler_class = torch.optim.lr_scheduler.__dict__[optimizer_config.scheduler.name]
        scheduler = scheduler_class(optimizer, **optimizer_config.scheduler.kwargs)
        logger.info(f"Created {optimizer_config.scheduler.name} scheduler")
    else:
        scheduler = None

    return optimizer, scheduler, total_params_in_optimizer
