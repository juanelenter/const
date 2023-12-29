#!/bin/bash

WANDB_MODE="online"
EPOCHS=200

MODEL_CONFIG="model=CIFAR_ResNet18"
DATA_CONFIG="data=cifar10_lt"
RESOURCES="cluster=local tasks_per_node=1 use_ddp=False"
# RESOURCES="cluster=long tasks_per_node=1 timeout_min=5 use_ddp=False"
WANDB_TAGS='("cifar_sweep",)'
seed=1

RESTART_ON_FEASIBLE="False"

SCHEDULER_CONFIG="optim.primal_optimizer.scheduler.name=CosineAnnealingLR optim.primal_optimizer.scheduler.kwargs.T_max=${EPOCHS}"
BASE_SGD_CONFIG="optim.primal_optimizer.name=SGD optim.primal_optimizer.shared_kwargs.momentum=0.9 optim.primal_optimizer.shared_kwargs.weight_decay=5e-3 ${SCHEDULER_CONFIG}"

IMBALANCES=(1.0)
POINTWISE_PROBABILITIES=(0.7 0.8 0.9 0.95)
EARLY_STOP_CHOICES=("False")

DUAL_LRATES=(0.01 0.02 0.04)
PRIMAL_LRATES=(0.01)
DATA_AUGMENTATION_CHOICES=("True" "False")

for data_aug in "${DATA_AUGMENTATION_CHOICES[@]}"
do
    for early_stop in "${EARLY_STOP_CHOICES[@]}"
    do
    for dual_lr in "${DUAL_LRATES[@]}"
        do
            for pointwise_probability in "${POINTWISE_PROBABILITIES[@]}"
            do
                for imbalance in "${IMBALANCES[@]}"
                do
                    for primal_lr in "${PRIMAL_LRATES[@]}"
                    do
                        DUAL_CONFIG="task.multiplier_kwargs.restart_on_feasible=${RESTART_ON_FEASIBLE} optim.dual_optimizer.shared_kwargs.lr=${dual_lr} task.early_stop_on_feasible=${early_stop}"
                        OPTIM_CONFIG="${BASE_SGD_CONFIG} optim.primal_optimizer.shared_kwargs.lr=${primal_lr}"
                        CUDA_VISIBLE_DEVICES=0 python main.py \
                        --model_config=configs/model.py:"${MODEL_CONFIG} ${OPTIM_CONFIG}" \
                        --data_config=configs/data.py:"${DATA_CONFIG} data.imbalance.kwargs.ratio=${imbalance} data.imbalance.seed=${seed} data.dataloader.seed=${seed} data.dataset_kwargs.use_data_augmentation=${data_aug}" \
                        --task_config=configs/task.py:"task=feasibility task.pointwise_probability=${pointwise_probability} ${DUAL_CONFIG}" \
                        --config.train.total_epochs=${EPOCHS} \
                        --metrics_config=configs/metrics.py:classification \
                        --resources_config=configs/resources.py:"${RESOURCES}" \
                        --config.logging.wandb_mode="${WANDB_MODE}" --config.logging.wandb_tags=${WANDB_TAGS} \
                        --config.train.seed=${seed}
                    done
                done
            done
        done
    done
done