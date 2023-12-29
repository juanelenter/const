#!/bin/bash

WANDB_MODE="online"
EPOCHS=30

MODEL_CONFIG="model=MNIST_ResNet18 optim.primal_optimizer.name=Adam optim.primal_optimizer.shared_kwargs.lr=1e-3"
DATA_CONFIG="data=mnist"
METRICS_CONFIG="classification"
RESOURCES="cluster=long tasks_per_node=1 timeout_min=30 use_ddp=False"
WANDB_TAGS='("mnist_sweep",)'

python main.py \
--model_config=configs/model.py:"${MODEL_CONFIG}" \
--data_config=configs/data.py:"${DATA_CONFIG}" \
--task_config=configs/task.py:"task=erm task.pointwise_probability=1.0" \
--config.train.total_epochs=${EPOCHS} \
--metrics_config=configs/metrics.py:"${METRICS_CONFIG}" \
--resources_config=configs/resources.py:"${RESOURCES}" \
--config.logging.wandb_mode="${WANDB_MODE}" --config.logging.wandb_tags=${WANDB_TAGS}

# POINTWISE_PROBABILITIES=(0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95)
POINTWISE_PROBABILITIES=(0.7 0.8 0.9)
EARLY_STOP_CHOICES=("False")

# No dual restarts. Decided to try on less learning rates
DUAL_LRS=(0.001)
IS_RESTART="False"

# # Dual restarts
# DUAL_LRS=(0.001)
# IS_RESTART="True"

for dual_lr in "${DUAL_LRS[@]}"
do
    for pointwise_probability in "${POINTWISE_PROBABILITIES[@]}"
    do
        for early_stop in "${EARLY_STOP_CHOICES[@]}"
        do
            DUAL_CONFIG="task.multiplier_kwargs.restart_on_feasible=${IS_RESTART} optim.dual_optimizer.shared_kwargs.lr=${dual_lr} task.early_stop_on_feasible=${early_stop}"

            python main.py \
            --model_config=configs/model.py:"${MODEL_CONFIG}" \
            --data_config=configs/data.py:"${DATA_CONFIG}" \
            --task_config=configs/task.py:"task=feasibility task.pointwise_probability=${pointwise_probability} ${DUAL_CONFIG}" \
            --config.train.total_epochs=${EPOCHS} \
            --metrics_config=configs/metrics.py:classification \
            --resources_config=configs/resources.py:"${RESOURCES}" \
            --config.logging.wandb_mode="${WANDB_MODE}" --config.logging.wandb_tags=${WANDB_TAGS}
        done
    done
done
