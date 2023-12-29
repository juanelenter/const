from types import SimpleNamespace

import cooper

import shared
from src.utils.optim import build_optimizer_from_config

logger = shared.fetch_main_logger()

CooperOptimizer = cooper.optim.ConstrainedOptimizer | cooper.optim.UnconstrainedOptimizer


def build_cooper_optimizer_and_schedulers(model, cmp, config) -> tuple[CooperOptimizer, SimpleNamespace]:
    primal_optimizer, primal_scheduler, num_params_in_primal_optimizer = build_optimizer_from_config(
        model, config.optim.primal_optimizer
    )
    logger.info(f"Instantiated primal optimizer: \n {primal_optimizer}")
    num_primal_params = sum([param.numel() for param in model.parameters()])
    logger.info(
        f"Created optimizer accounts for {num_params_in_primal_optimizer}/{num_primal_params} primal parameters"
    )

    if cmp.has_dual_variables:
        dual_optimizer, dual_scheduler = _build_dual_optimizer_from_config(config.optim.dual_optimizer, cmp)
        cooper_optimizer_kwargs = {"dual_optimizers": dual_optimizer, "constraint_groups": cmp.constraint_groups}
    else:
        dual_scheduler = None
        cooper_optimizer_kwargs = {}

    schedulers = SimpleNamespace(primal=primal_scheduler, dual=dual_scheduler)

    if config.optim.cooper_optimizer_name == "ExtrapolationConstrainedOptimizer" and config.resources.use_ddp:
        # This is because we manually sync the multipliers after every step (see
        # `Trainer._sync_feasibility_multiplier`). However, the extrapolation optimizer
        # would internally perform an update to the multipliers and we are not tackling
        # the sync at the extrapolation point.
        # Most likely we will not be using extrapolation in our experiments, so this
        # should not be a big problem.
        raise ValueError("The current implementation does not support DDP with `ExtrapolationConstrainedOptimizer`")

    cooper_optimizer_class = getattr(cooper.optim, config.optim.cooper_optimizer_name)
    cooper_optimizer = cooper_optimizer_class(primal_optimizer, **cooper_optimizer_kwargs)
    logger.info(f"Created Cooper optimizer: \n {cooper_optimizer}")

    return cooper_optimizer, schedulers


def _build_dual_optimizer_from_config(optimizer_config, cmp):
    # As dual variables aim to *maximize* the Lagrangian, we hard-code `maximize=True`

    with optimizer_config.unlocked():
        optimizer_config.shared_kwargs["maximize"] = True

    assert len(cmp.constraint_groups) == 1, "Code below assumes that only the feasibility constraint is present"
    multiplier = cmp.feasibility_constraint.multiplier

    optimizer, scheduler, num_params_in_optimizer = build_optimizer_from_config(
        model=None, optimizer_config=optimizer_config, parameter_groups=cmp.dual_parameter_groups()
    )
    logger.info(f"Instantiated dual optimizer: \n {optimizer}")

    num_dual_params = [sum([_.numel() for _ in group]) for group in cmp.dual_parameter_groups().values()]
    num_dual_params = sum(num_dual_params)
    logger.info(f"Created optimizer account for {num_params_in_optimizer}/{num_dual_params} dual parameters")

    return optimizer, scheduler
