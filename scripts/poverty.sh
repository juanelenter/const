#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
 
WANDB_MODE="online"
EPOCHS=200

# MODEL_CONFIG="model=MNIST_ResNet18 optim.primal_optimizer.name=Adam optim.primal_optimizer.shared_kwargs.lr=1e-4"
MODEL_CONFIG="model=Poverty_ResNet18"
RESOURCES="cluster=debug tasks_per_node=1 use_ddp=False"
# RESOURCES="cluster=long tasks_per_node=1 timeout_min=5 use_ddp=False"
OPTIM_CONFIG="optim.primal_optimizer.name=Adam optim.primal_optimizer.shared_kwargs.weight_decay=0 optim.primal_optimizer.shared_kwargs.lr=1e-3 optim.primal_optimizer.scheduler.name=StepLR optim.primal_optimizer.scheduler.kwargs.gamma=0.96 optim.primal_optimizer.scheduler.kwargs.step_size=1"

POINTWISE_LOSS=(0.3 0.5 0.7)
SEEDS=(44 55 66)
DUALLRS=(0.001 0.01)

for DUALLR in "${DUALLRS[@]}"
do
    DUAL_CONFIG="task.multiplier_kwargs.restart_on_feasible=False optim.dual_optimizer.shared_kwargs.lr=${DUALLR} task.early_stop_on_feasible=False"
    for SEED in "${SEEDS[@]}"
    do
        for pointwise_loss in "${POINTWISE_LOSS[@]}"
        do
            DATA_CONFIG="data=poverty data.dataloader.seed=${SEED}"
            python main.py \
                --model_config=configs/model.py:"${MODEL_CONFIG} ${OPTIM_CONFIG}" \
                --data_config=configs/data.py:"${DATA_CONFIG}" \
                --task_config=configs/task.py:"task=feasibility task.pointwise_loss=${pointwise_loss} ${DUAL_CONFIG}" \
                --config.train.total_epochs=${EPOCHS} \
                --metrics_config=configs/metrics.py:regression \
                --resources_config=configs/resources.py:"${RESOURCES}" \
                --config.logging.wandb_mode="${WANDB_MODE}" \
                --config.train.seed=${SEED}
        done
    done
done






