import os
import socket
from types import SimpleNamespace
from typing import Any, Union

import submitit
import torch
import torch.distributed as dist

import shared

logger = shared.fetch_main_logger()


def get_num_workers() -> int:
    """Gets the optimal number of DatLoader workers to use in the current job."""

    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return torch.multiprocessing.cpu_count()


def init_distributed(resources_config):
    """
    Initialize torch.distributed if using multiple GPUs.
    Extracts relevant info from SLURM environment variables.
    Sets device based on local_rank.
    Defines {multi_gpu, rank, local_rank, world_size, device}
    """

    num_workers = get_num_workers()

    if (resources_config.nodes * resources_config.tasks_per_node) > 1:
        logger.info("Initializing multi-GPU environment")

        multi_gpu = True

        if not dist.is_initialized():
            job_env = submitit.JobEnvironment()
            local_rank = job_env.local_rank
            rank = job_env.global_rank
            world_size = job_env.num_tasks
            logger.info(f"Initialized process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")

            os.environ["MASTER_ADDR"] = os.getenv("SLURM_NODELIST")
            os.environ["MASTER_PORT"] = "40101"

            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

            dist.init_process_group(backend="nccl", init_method="env://", rank=local_rank, world_size=world_size)
            torch.distributed.barrier()

        logger.info("torch.distributed is initialized")

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Ignoring this block for now since we don't use 8+ GPUs
        # if world_size > 8:
        #     assert slurm_local_rank >= 0
        #     local_rank = slurm_local_rank
        # else:
        #     local_rank = rank
    else:
        if torch.cuda.is_available():
            logger.info("This is a single GPU job")
        else:
            logger.info("This is a CPU job")
        multi_gpu = False
        world_size = 1
        rank = 0
        local_rank = 0

    logger.info(f"Rank {rank}")
    logger.info(f"World size {world_size}")
    logger.info(f"Local rank {local_rank}")
    logger.info(f"Running on host {socket.gethostname()}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    return SimpleNamespace(
        multi_gpu=multi_gpu,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        num_workers=num_workers,
    )


def wait_for_all_processes():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def do_all_gather_object(object):
    if dist.is_available() and dist.is_initialized():
        output_objects = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(output_objects, object)
        return output_objects
    return [object]


def do_broadcast_object(object):
    objects = [object]
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(objects)
    return objects[0]


def do_reduce_mean(data: Union[list[dict[str, Any]], dict[str, Any], torch.Tensor, int], dst_rank: int = 0):
    if not dist.is_available() or not dist.is_initialized():
        return data

    if isinstance(data, torch.Tensor):
        cloned_tensor = data.clone()
        dist.reduce(cloned_tensor, dst=dst_rank, op=dist.ReduceOp.SUM)
        return cloned_tensor / dist.get_world_size()
    elif isinstance(data, int):
        # The original data was integers on each worker, so we extract the value from
        # the tensor to avoid returning a tensor object (as opposed to a float).
        return do_reduce_mean(torch.tensor(data, dtype=torch.int, device="cuda"), dst_rank=dst_rank).item()
    elif isinstance(data, dict):
        return {k: do_reduce_mean(v, dst_rank=dst_rank) for k, v in data.items()}
    elif isinstance(data, list):
        return [do_reduce_mean(v, dst_rank=dst_rank) for v in data]
    else:
        raise ValueError(f"Unsupported type: {type(data)}")
