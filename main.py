import logging
import os
import sys
from datetime import datetime

import cooper
import dotenv
import submitit
import torch
from absl import app
from absl.flags import FLAGS
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags as MLC_FLAGS

import shared
from src.trainer import Trainer

# Initialize "FLAGS.config" object with a placeholder config
MLC_FLAGS.DEFINE_config_file("config", default="configs/core.py")
MLC_FLAGS.DEFINE_config_file("model_config", default="configs/model.py")
MLC_FLAGS.DEFINE_config_file("data_config", default="configs/data.py")
MLC_FLAGS.DEFINE_config_file("task_config", default="configs/task.py")
MLC_FLAGS.DEFINE_config_file("metrics_config", default="configs/metrics.py")
MLC_FLAGS.DEFINE_config_file("resources_config", default="configs/resources.py")


logger = logging.getLogger()
# Configure root logger to use rich logging
shared.configure_logger(logger, level=logging.INFO)

# Load environment variables from .env file. This file is not tracked by git.
dotenv.load_dotenv()


def inject_file_configs(config, injected_config_fields):
    for field_name in injected_config_fields:
        logger.info(f"Injecting field `{field_name}` in config")
        existing_config = getattr(config, field_name, {})
        injected_config = getattr(FLAGS, f"{field_name}_config", {})

        if len(injected_config.keys()) > 0:
            if len(existing_config.keys()) > 0:
                raise RuntimeError(f"Cannot specify both `config.{field_name}` and `{field_name}_config`")

            logger.info(f"Using {field_name} config from `{field_name}_config` provided in CLI")
            for key_from_root, value in injected_config.items():
                with config.unlocked():
                    if key_from_root not in config:
                        shared.drill_to_key_and_set(config, key_from_root, value)
                    elif isinstance(config[key_from_root], (dict, ConfigDict)):
                        config[key_from_root].update(value)
                    else:
                        raise RuntimeError(f"Cannot inject `{field_name}_config` into `{key_from_root}`")

    return config


def main(_):
    logger.info(f"Using Python version {sys.version}")
    logger.info(f"Using PyTorch version {torch.__version__}")
    logger.info(f"Using Cooper version {cooper.__version__}")

    injected_config_fields = ["model", "data", "task", "metrics", "resources"]
    config = inject_file_configs(FLAGS.config, injected_config_fields)

    # TODO: Do we want to use RsyncSnapshot to get a snapshot of the code?
    # with submitit.helpers.RsyncSnapshot(snapshot_dir=Path(os.environ["SUBMITIT_DIR"]), with_submodules=True):
    logger.info(f"Current working directory: {os.getcwd()}")

    # Get a directory name with current date and time
    job_submitit_dir = os.path.join(os.environ["SUBMITIT_DIR"], datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # Queue a the job. If `cluster` is not SLURM, the slurm-exclusive parameters will be ignored.
    executor = submitit.AutoExecutor(cluster=config.resources.cluster, folder=job_submitit_dir)
    executor.update_parameters(
        name=config.exp_name,
        slurm_mem=config.resources.mem,
        slurm_partition=config.resources.partition,
        timeout_min=config.resources.timeout_min,
        slurm_gpus_per_node=config.resources.tasks_per_node,
        slurm_ntasks_per_node=config.resources.tasks_per_node,
        cpus_per_task=config.resources.cpus_per_task,
        nodes=config.resources.nodes,
        slurm_comment=config.resources.comment,
        slurm_exclude=config.resources.exclude,
    )

    logger.info("Instantiating Trainer from config")
    trainer = Trainer(config)

    job = executor.submit(trainer)
    logger.info(f"Submitted experiment with jobid {job.job_id}")

    if config.resources.cluster == "debug":
        job.result()


if __name__ == "__main__":
    app.run(main)
