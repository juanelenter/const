#!/bin/bash

WANDB_MODE="online"
WANDB_PROJECT="feasible-learning"
EPOCHS=60

# MODEL_CONFIG="model=MNIST_ResNet18 optim.primal_optimizer.name=Adam optim.primal_optimizer.shared_kwargs.lr=1e-4"
MODEL_CONFIG="model=MNIST_CNN optim.primal_optimizer.name=Adam optim.primal_optimizer.shared_kwargs.lr=1e-4"
DATA_CONFIG="data=mnist"
METRICS_CONFIG="classification"
RESOURCES="cluster=debug tasks_per_node=1 use_ddp=False"
# RESOURCES="cluster=long tasks_per_node=1 timeout_min=5 use_ddp=False"


POINTWISE_PROBABILITIES=(0.9)
# EARLY_STOP_CHOICES=("True" "False")
EARLY_STOP_CHOICES=("True")


# python main.py \
# --model_config=configs/model.py:"${MODEL_CONFIG}" \
# --data_config=configs/data.py:"${DATA_CONFIG}" \
# --task_config=configs/task.py:"task=erm task.pointwise_probability=1.0" \
# --config.train.total_epochs=${EPOCHS} \
# --metrics_config=configs/metrics.py:"${METRICS_CONFIG}" \
# --resources_config=configs/resources.py:"${RESOURCES}" \
# --config.logging.wandb_mode="${WANDB_MODE}"

for pointwise_probability in "${POINTWISE_PROBABILITIES[@]}"
do

    for early_stop in "${EARLY_STOP_CHOICES[@]}"
    do
        # PI_CONFIG="optim.dual_optimizer.name=PI optim.dual_optimizer.shared_kwargs.lr=1e-3 optim.dual_optimizer.shared_kwargs.Kp=0.0 optim.dual_optimizer.shared_kwargs.Ki=1.0"
        PI_CONFIG="optim.dual_optimizer.name=PI optim.dual_optimizer.shared_kwargs.lr=1e-3 optim.dual_optimizer.shared_kwargs.Kp=1.0 optim.dual_optimizer.shared_kwargs.Ki=1.0"
        # PI_CONFIG="optim.dual_optimizer.name=PI optim.dual_optimizer.shared_kwargs.lr=1e-3 optim.dual_optimizer.shared_kwargs.Kp=5.0 optim.dual_optimizer.shared_kwargs.Ki=1.0"
        DUAL_CONFIG="task.multiplier_kwargs.restart_on_feasible=False ${PI_CONFIG} task.early_stop_on_feasible=${early_stop}"
        # DUAL_CONFIG="task.multiplier_kwargs.restart_on_feasible=False ${PI_CONFIG} task.early_stop_on_feasible=${early_stop}"

        python main.py \
        --model_config=configs/model.py:"${MODEL_CONFIG}" \
        --data_config=configs/data.py:"${DATA_CONFIG}" \
        --task_config=configs/task.py:"task=feasibility task.pointwise_probability=${pointwise_probability} ${DUAL_CONFIG}" \
        --config.train.total_epochs=${EPOCHS} \
        --metrics_config=configs/metrics.py:classification \
        --resources_config=configs/resources.py:"${RESOURCES}" \
        --config.logging.wandb_mode="${WANDB_MODE}"  --config.logging.wandb_project="${WANDB_PROJECT}"
    done
done
