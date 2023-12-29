#!/bin/bash

WANDB_MODE="online"
EPOCHS=5000

DATA_CONFIG="data=two_moons data.dataset_kwargs.train_samples=100 data.dataset_kwargs.val_samples=2000"
RESOURCES="cluster=debug tasks_per_node=1 use_ddp=False"
# RESOURCES="cluster=long tasks_per_node=1 timeout_min=5 use_ddp=False"

HIDDEN_SIZES=(100) # 50 100 200 400)

# To run "pure" ERM, you can set task.pointwise_probability=1.0
TASKS=("erm" "feasibility")
POINTWISE_PROBABILITIES=(0.9 0.95)



for hidden_size in "${HIDDEN_SIZES[@]}"
do
    # Run ERM version
    python main.py \
    --model_config=configs/model.py:"model=TwoDim_MLP model.init_kwargs.hidden_sizes=(${hidden_size},)" \
    --data_config=configs/data.py:"${DATA_CONFIG}" \
    --task_config=configs/task.py:"task=erm" \
    --config.train.total_epochs=${EPOCHS} \
    --metrics_config=configs/metrics.py:classification \
    --resources_config=configs/resources.py:"${RESOURCES}" \
    --config.logging.wandb_mode="${WANDB_MODE}"

    for pointwise_probability in "${POINTWISE_PROBABILITIES[@]}"
    do
        python main.py \
        --model_config=configs/model.py:"model=TwoDim_MLP model.init_kwargs.hidden_sizes=(${hidden_size},)" \
        --data_config=configs/data.py:"${DATA_CONFIG}" \
        --task_config=configs/task.py:"task=feasibility task.pointwise_probability=${pointwise_probability} task.multiplier_kwargs.restart_on_feasible=False" \
        --config.train.total_epochs=${EPOCHS} \
        --metrics_config=configs/metrics.py:classification \
        --resources_config=configs/resources.py:"${RESOURCES}" \
        --config.logging.wandb_mode="${WANDB_MODE}"

    done
done
