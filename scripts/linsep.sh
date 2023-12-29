#!/bin/bash

WANDB_MODE="online"
EPOCHS=500

PRIMAL_CONFIG="optim.primal_optimizer.name=Adam optim.primal_optimizer.shared_kwargs.lr=7e-1 optim.primal_optimizer.shared_kwargs.weight_decay=0.0"
DATA_CONFIG="data=linsep_2d data.dataset_kwargs.train_samples=128 data.dataset_kwargs.val_samples=1000"
RESOURCES="cluster=debug tasks_per_node=1 use_ddp=False"
# RESOURCES="cluster=long tasks_per_node=1 timeout_min=5 use_ddp=False"

HIDDEN_SIZES=("()" "(1,)" "(2,)" "(5,)" "(10,)")

POINTWISE_PROBABILITIES=(0.99)
EARLY_STOP_CHOICES=("True" "False")


for hidden_size in "${HIDDEN_SIZES[@]}"
do

    python main.py \
    --model_config=configs/model.py:"model=TwoDim_MLP model.init_kwargs.hidden_sizes=${hidden_size} ${PRIMAL_CONFIG}" \
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
            DUAL_CONFIG="task.multiplier_kwargs.restart_on_feasible=False optim.dual_optimizer.shared_kwargs.lr=1e-3 task.early_stop_on_feasible=${early_stop}"

            python main.py \
            --model_config=configs/model.py:"model=TwoDim_MLP model.init_kwargs.hidden_sizes=${hidden_size} ${PRIMAL_CONFIG}" \
            --data_config=configs/data.py:"${DATA_CONFIG}" \
            --task_config=configs/task.py:"task=feasibility task.pointwise_probability=${pointwise_probability} ${DUAL_CONFIG}" \
            --config.train.total_epochs=${EPOCHS} \
            --metrics_config=configs/metrics.py:classification \
            --resources_config=configs/resources.py:"${RESOURCES}" \
            --config.logging.wandb_mode="${WANDB_MODE}"
        done
    done
done
