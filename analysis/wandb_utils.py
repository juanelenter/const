import logging
import multiprocessing
import os
import pickle
import sys
import tempfile

import dotenv
import torch
import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import shared
from src import models

logger = logging.getLogger()
# Configure root logger to use rich logging
shared.configure_logger(logger, level=logging.INFO)

dotenv.load_dotenv()

logging.basicConfig()
logger = logging.getLogger()

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.benchmark = False


# TODO(juan43ramirez): docstrings, please!


def fetch_models_and_configs(run_ids: list[str], entity: str = None, project: str = None):
    if entity is None:
        entity = os.environ["WANDB_ENTITY"]

    if project is None:
        raise ValueError("WandB project must be specified since we use multiple projects in this entity")

    api = wandb.Api(overrides={"entity": entity, "project": project}, timeout=20)

    if "SLURM_JOB_ID" in os.environ:
        tmp_dir_path = os.environ["SLURM_TMPDIR"]
    else:
        tmp_dir_path = tempfile.TemporaryDirectory().name
    os.makedirs(tmp_dir_path, exist_ok=True)

    configs, models = {}, {}
    for run_id in run_ids:
        run = api.run(path=f"{entity}/{project}/{run_id}")
        configs[run_id] = load_config(run, tmp_dir_path)
        models[run_id] = load_model(run, configs[run_id], tmp_dir_path)

    return configs, models


def load_model(run, config, tmp_dir_path):
    logger.info(f"Loading model from run {run.id}")
    checkpoint_file = run.file("checkpoint.pt").download(root=tmp_dir_path, replace=True)
    checkpoint = torch.load(checkpoint_file.name, map_location=DEVICE)
    state_dict = checkpoint["model"]

    model_class = getattr(models, config.model.name)
    model = model_class(**config.model.init_kwargs)
    model.load_state_dict(state_dict)
    logger.info("Done loading model")

    return model


def load_config(run, tmp_dir_path):
    logger.info(f"Loading config from run {run.id}")
    run.file("config.pkl").download(root=tmp_dir_path, replace=True)
    with open(os.path.join(tmp_dir_path, "config.pkl"), "rb") as pickle_file:
        config = pickle.load(pickle_file)

    config = patch_config_for_current_system(config)

    logger.info("Done loading config")

    return config


def load_metrics(run, metric_keys, _x_axis="_epoch"):
    # history subsamples the metrics by default, so we set samples to a large number
    metric = run.history(keys=metric_keys, samples=1000000, x_axis=_x_axis)

    return metric


def load_history_from_files(run, config, tmp_dir_path, metric, split=""):
    history = []
    path = f"{metric}"
    if split != "":
        path += f"/{split}"
    os.makedirs(os.path.join(tmp_dir_path, path), exist_ok=True)

    for epoch in range(config.train.total_epochs):
        filename = f"{path}/epoch_{epoch}.pt"
        run.file(filename).download(root=tmp_dir_path, replace=True)
        epoch_values = torch.load(os.path.join(tmp_dir_path, filename))
        history.append(epoch_values)

    history = torch.stack(history).squeeze()

    return history


def patch_config_for_current_system(config):
    config.resources.tasks_per_node = 1
    config.resources.use_ddp = False
    config.resources.cpus_per_task = multiprocessing.cpu_count() - 1

    config.data.dataset_kwargs.data_path = os.environ["DATA_DIR"]

    return config


if __name__ == "__main__":
    run_id = "3549679"
    wandb_entity = os.environ["WANDB_ENTITY"]
    wandb_project = "feasible-learning"
    api = wandb.Api(overrides={"entity": wandb_entity, "project": wandb_project}, timeout=20)
    run = api.run(path=f"{wandb_entity}/{wandb_project}/{run_id}")
    tmp_dir_path = tempfile.TemporaryDirectory().name
    os.makedirs(tmp_dir_path, exist_ok=True)

    config = load_config(run, tmp_dir_path)
    model = load_model(run, config, tmp_dir_path)

    epoch_metrics = load_metrics(run, ["train/violation/max"], _x_axis="_epoch")
    batch_metrics = load_metrics(run, ["batch/avg_loss", "batch/avg_acc"], _x_axis="_step")

    multiplier_history = load_history_from_files(run, config, tmp_dir_path, "multipliers")
    train_loss_history = load_history_from_files(run, config, tmp_dir_path, "losses", split="train")
    val_loss_history = load_history_from_files(run, config, tmp_dir_path, "losses", split="val")

    breakpoint()

    tempfile.TemporaryDirectory().cleanup()
