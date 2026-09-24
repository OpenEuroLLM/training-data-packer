import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from loguru import logger

from training_data_packer.metadata import Metadata


def _get_my_partition_tasks(files: list[Any], task_count: int, task_id: int) -> list[Any]:
    size, rest = divmod(len(files), task_count)
    grouped_files = [files[i * size + min(i, rest) : (i + 1) * size + min(i + 1, rest)] for i in range(task_count)]
    return grouped_files[task_id - 1]


def get_my_slurm_tasks(files: list[Any]) -> list[Any]:
    """Get selection of files to be processed by current slurm array task id.

    Args:
        files: List of files to be partitioned.

    Returns:
        The subset of files to be processed by current slurm array task id.

    """
    task_count = os.environ["SLURM_ARRAY_TASK_COUNT"]
    task_id = os.environ["SLURM_ARRAY_TASK_ID"]
    task_files = _get_my_partition_tasks(files, int(task_count), int(task_id))
    logger.info(f"Slurm task id: {task_id} of {task_count}, processing {len(task_files)} files")
    return task_files


def schedule_files(
    all_files: list[os.PathLike],
    metadata: Metadata,
    function: Callable[[os.PathLike], None],
    workers: int = 1,
    slurm: bool = False,
):
    """Schedule files for processing.

    Schedules the execution of a specific function across a list of files,
    supporting both sequential and parallel execution modes as well as integration
    with SLURM job arrays.

    This function determines the set of files to process based on the provided
    parameters. If the SLURM flag is enabled, it filters the input list to match
    the specific tasks assigned to the current job step; otherwise, it processes
    all provided files. The execution strategy depends on the number of workers
    specified: a count greater than one utilizes a process pool to distribute
    tasks in parallel, while a count of one processes files sequentially. In the
    event of parallel execution, the function monitors the worker processes and
    raises a collective error if any individual task fails.

    Args:
        all_files: List of file paths that are candidates for processing.
        metadata: Metadata configuration.
        function: Callable that accepts a single file path argument and performs
            the desired operations.
        workers: Number of parallel worker processes to utilize. If set to 1,
            processing occurs sequentially. Defaults to 1.
        slurm: Flag indicating whether to filter files based on the current SLURM
            task ID. Defaults to False.

    Raises:
        RuntimeError: If the workers parameter is greater than one and one or
            more worker processes fail during execution.

    Returns:
        None

    """
    if slurm:
        task_files = get_my_slurm_tasks(all_files)
    else:
        logger.info("Not a SLURM task, processing all files")
        task_files = all_files

    if workers > 1:
        jobs = []
        fail = False
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for src_file in task_files:
                job = executor.submit(function, src_file, metadata)
                jobs.append(job)
            executor.shutdown(wait=True)
        for n, job in enumerate(jobs):
            if job.exception() is not None:
                logger.error(f"There were an exception thrown for file {task_files[n]}: {job.exception()}")
                fail = True
        if fail:
            raise RuntimeError("One or more workers failed")
    else:
        for src_file in task_files:
            logger.debug(f"Processing file {src_file}")
            function(src_file, metadata)
