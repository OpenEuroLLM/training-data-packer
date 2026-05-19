import os
from typing import Any

from loguru import logger


def _get_my_partition_tasks(files: list[Any], task_count: int, task_id: int) -> list[Any]:
    size, rest = divmod(len(files), task_count)
    grouped_files = [files[i * size + min(i, rest) : (i + 1) * size + min(i + 1, rest)] for i in range(task_count)]
    return grouped_files[task_id - 1]


def get_my_slurm_tasks(files: list[Any]) -> list[Any]:
    task_count = os.environ["SLURM_ARRAY_TASK_COUNT"]
    task_id = os.environ["SLURM_ARRAY_TASK_ID"]
    task_files = _get_my_partition_tasks(files, int(task_count), int(task_id))
    logger.info(f"Slurm task id: {task_id} of {task_count}, processing {len(task_files)} files")
    return task_files
