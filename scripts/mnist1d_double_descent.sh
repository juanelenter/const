#!/bin/bash

# Script for running the MNIST1D experiment with an MLP model
# We compare the effect of different model sizes and target pointwise probabilities
# when training on the ERM and feasibility formulations

# To run "pure" ERM, you can set task.pointwise_probability=1.0
RESOURCES="cluster=debug tasks_per_node=1 use_ddp=False"
# RESOURCES="cluster=long tasks_per_node=1 timeout_min=5 use_ddp=False"

TASKS=("erm" "feasibility")
HIDDEN_SIZES=(30 50 100 200 400)
POINTWISE_PROBABILITIES=(0.7 0.8 0.9 0.95)

for task in "${TASKS[@]}"
do
    for hidden_size in "${HIDDEN_SIZES[@]}"
    do
        for pointwise_probability in "${POINTWISE_PROBABILITIES[@]}"
        do
            python main.py \
                --model_config=configs/model.py:"model=MNIST1D_MLP model.init_kwargs.hidden_sizes=(${hidden_size},)" \
                --data_config=configs/data.py:"mnist1d" \
                --task_config=configs/task.py:"task=${task} task.pointwise_probability=${pointwise_probability}" \
                --config.train.total_epochs=200 \
                --metrics_config=configs/metrics.py:classification \
                --resources_config=configs/resources.py:"${RESOURCES}" \
                --config.logging.wandb_mode=online
        done
    done
done
