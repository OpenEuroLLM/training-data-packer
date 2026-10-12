import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from loguru import logger

from training_data_packer.metadata import Metadata


def _get_my_partition_tasks(files: list[Any], task_count: int, task_id: int) -> list[Any]:
    return files[task_id - 1 :: task_count]


def get_my_slurm_tasks(files: list[Any]) -> list[Any]:
    """Get selection of files to be processed by current slurm array task id.

    :param files: files to partition
    :returns: the subset of files to be processed by the current slurm array task
    """
    task_count = os.environ["SLURM_ARRAY_TASK_COUNT"]
    task_id = os.environ["SLURM_ARRAY_TASK_ID"]
    task_files = _get_my_partition_tasks(files, int(task_count), int(task_id))
    logger.info(f"Slurm task id: {task_id} of {task_count}, processing {len(task_files)} files")
    return task_files


def schedule_files(
    all_work_items: list[os.PathLike | str],
    metadata: Metadata,
    function: Callable[[os.PathLike], None],
    workers: int = 1,
    slurm: bool = False,
):
    """Schedule files for processing.

    Run a function on each work item, sequentially or in parallel. If ``slurm`` is set,
    only the items assigned to the current SLURM array task are processed. With more
    than one worker, items are distributed over a process pool.

    :param all_work_items: file paths, part names, etc. that are candidates for processing
    :param metadata: metadata of the collection
    :param function: function called with a work item and the metadata
    :param workers: number of parallel worker processes; 1 means sequential processing
    :param slurm: process only the items assigned to the current SLURM array task
    :raises RuntimeError: if one or more worker processes fail
    """
    if slurm:
        task_work_items = all_work_items
    else:
        logger.info("Not a SLURM task, processing all files")
        task_work_items = all_work_items

    if workers > 1:
        jobs = []
        fail = False
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for work_item in task_work_items:
                job = executor.submit(function, work_item, metadata)
                jobs.append(job)
            executor.shutdown(wait=True)
        for n, job in enumerate(jobs):
            if job.exception() is not None:
                logger.error(f"There were an exception thrown for file {task_work_items[n]}: {job.exception()}")
                fail = True
        if fail:
            raise RuntimeError("One or more workers failed")
    else:
        for work_item in task_work_items:
            logger.debug(f"Processing file {work_item}")
            function(work_item, metadata)
