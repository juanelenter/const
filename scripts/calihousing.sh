#!/bin/bash

WANDB_MODE="online"
EPOCHS=2000

DATA_CONFIG="data=calihousing"
RESOURCES="cluster=debug tasks_per_node=1 use_ddp=False"
# RESOURCES="cluster=long tasks_per_node=1 timeout_min=5 use_ddp=False"

HIDDEN_SIZES=("(64,32,16,8,)")
LRS=(0.0001)
POINTWISE_LOSSES=(0.3 0.2 0.5)

for lr in "${LRS[@]}"
do
    PRIMAL_CONFIG="optim.primal_optimizer.name=Adam optim.primal_optimizer.shared_kwargs.lr=${lr} optim.primal_optimizer.shared_kwargs.weight_decay=0.0"
    for hidden_size in "${HIDDEN_SIZES[@]}"
    do
        for pointwise_loss in "${POINTWISE_LOSSES[@]}"
            do
                DUAL_CONFIG="task.multiplier_kwargs.restart_on_feasible=False optim.dual_optimizer.shared_kwargs.lr=1e-2 task.early_stop_on_feasible=False"

                python main.py \
                    --model_config=configs/model.py:"model=CaliHousing_MLP model.init_kwargs.hidden_sizes=${hidden_size} ${PRIMAL_CONFIG}" \
                    --data_config=configs/data.py:"${DATA_CONFIG}" \
                    --task_config=configs/task.py:"task=feasibility task.pointwise_loss=${pointwise_loss} ${DUAL_CONFIG}" \
                    --config.train.total_epochs=${EPOCHS} \
                    --metrics_config=configs/metrics.py:regression \
                    --resources_config=configs/resources.py:"${RESOURCES}" \
                    --config.logging.wandb_mode="${WANDB_MODE}"
            done
        python main.py \
            --model_config=configs/model.py:"model=CaliHousing_MLP model.init_kwargs.hidden_sizes=${hidden_size} ${PRIMAL_CONFIG}" \
            --data_config=configs/data.py:"${DATA_CONFIG}" \
            --task_config=configs/task.py:"task=erm task.pointwise_loss=0.0" \
            --config.train.total_epochs=${EPOCHS} \
            --metrics_config=configs/metrics.py:regression \
            --resources_config=configs/resources.py:"${RESOURCES}" \
            --config.logging.wandb_mode="${WANDB_MODE}"
    done
done
