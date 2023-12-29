#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

WANDB_MODE="online"
EPOCHS=100

# MODEL_CONFIG="model=MNIST_ResNet18 optim.primal_optimizer.name=Adam optim.primal_optimizer.shared_kwargs.lr=1e-4"
MODEL_CONFIG="model=AudioCNN optim.primal_optimizer.name=Adam optim.primal_optimizer.shared_kwargs.lr=1e-2 optim.primal_optimizer.shared_kwargs.weight_decay=1e-4 model.init_kwargs.n_channel=64"
DATA_CONFIG="data=speechcommands"
RESOURCES="cluster=debug tasks_per_node=1 use_ddp=False"
# RESOURCES="cluster=long tasks_per_node=1 timeout_min=5 use_ddp=False"


POINTWISE_PROBABILITIES=(0.95 0.9 0.85 0.8 0.75 0.7 0.5 0.3)
# EARLY_STOP_CHOICES=("True" "False")
EARLY_STOP_CHOICES=("False")

python main.py \
--model_config=configs/model.py:"${MODEL_CONFIG}" \
--data_config=configs/data.py:"${DATA_CONFIG}" \
--task_config=configs/task.py:"task=erm task.pointwise_probability=1.0" \
--config.train.total_epochs=${EPOCHS} \
--metrics_config=configs/metrics.py:classification \
--resources_config=configs/resources.py:"${RESOURCES}" \
--config.logging.wandb_mode="${WANDB_MODE}"

for pointwise_probability in "${POINTWISE_PROBABILITIES[@]}"
do

    for early_stop in "${EARLY_STOP_CHOICES[@]}"
    do
        DUAL_CONFIG="task.multiplier_kwargs.restart_on_feasible=False optim.dual_optimizer.shared_kwargs.lr=1e-1 task.early_stop_on_feasible=${early_stop}"

        python main.py \
        --model_config=configs/model.py:"${MODEL_CONFIG}" \
        --data_config=configs/data.py:"${DATA_CONFIG}" \
        --task_config=configs/task.py:"task=feasibility task.pointwise_probability=${pointwise_probability} ${DUAL_CONFIG}" \
        --config.train.total_epochs=${EPOCHS} \
        --metrics_config=configs/metrics.py:classification \
        --resources_config=configs/resources.py:"${RESOURCES}" \
        --config.logging.wandb_mode="${WANDB_MODE}"
    done
done
