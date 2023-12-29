#!/bin/bash

# export NCCL_DEBUG=INFO
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=INFO

# Source: https://github.com/Lightning-AI/lightning/issues/4420#issuecomment-919478212
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

WANDB_MODE="online"
EPOCHS=200

METRICS_CONFIG="large_scale_classification"
MODEL_CONFIG="model=ImageNet_ResNet50"
DATA_CONFIG="data=imagenet"

RESOURCES="cluster=main use_ddp=True tasks_per_node=2 timeout_min=30"
# RESOURCES="cluster=main use_ddp=True tasks_per_node=1 timeout_min=30"
# RESOURCES="cluster=debug_main use_ddp=False tasks_per_node=1 timeout_min=30"
# RESOURCES="cluster=long tasks_per_node=1 timeout_min=5 use_ddp=False"


POINTWISE_PROBABILITIES=(0.9)
# EARLY_STOP_CHOICES=("True" "False")
EARLY_STOP_CHOICES=("True")


# python main.py \
#     --model_config=configs/model.py:"${MODEL_CONFIG}" \
#     --data_config=configs/data.py:"${DATA_CONFIG}" \
#     --task_config=configs/task.py:"task=erm task.pointwise_probability=1.0" \
#     --config.train.total_epochs=${EPOCHS} \
#     --metrics_config=configs/metrics.py:"${METRICS_CONFIG}" \
#     --resources_config=configs/resources.py:"${RESOURCES}" \
#     --config.logging.wandb_mode="${WANDB_MODE}"

for pointwise_probability in "${POINTWISE_PROBABILITIES[@]}"
do

    for early_stop in "${EARLY_STOP_CHOICES[@]}"
    do
        DUAL_CONFIG="task.multiplier_kwargs.restart_on_feasible=False optim.dual_optimizer.shared_kwargs.lr=1e-3 task.early_stop_on_feasible=${early_stop}"

        python main.py \
        --model_config=configs/model.py:"${MODEL_CONFIG}" \
        --data_config=configs/data.py:"${DATA_CONFIG}" \
        --task_config=configs/task.py:"task=feasibility task.pointwise_probability=${pointwise_probability} ${DUAL_CONFIG}" \
        --config.train.total_epochs=${EPOCHS} \
        --metrics_config=configs/metrics.py:"${METRICS_CONFIG}" \
        --resources_config=configs/resources.py:"${RESOURCES}" \
        --config.logging.wandb_mode="${WANDB_MODE}"
    done
done
