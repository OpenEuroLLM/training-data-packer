"""Plan SLURM array sizes and time limits for the package, merge and tokenize steps of the pipeline.

The plan is based on the files in the directory referred by ``release.default.input`` that are included in a
release part. The heuristics are:

* 1 GiB of input is processed per minute.
* A file, or for merge a part, can not be split between tasks.
* The package step processes 8 files in parallel in a task. Merge and tokenize process one at a time.
* A task should ideally run for 5 hours, but never longer than the partition allows.
* An array has at most 200 tasks in partition ``small`` and 100 tasks in partition ``standard``.

The estimated time is multiplied by a safety ratio before the plan is made, e.g. 1.5 reserves 50% more time than
estimated. The default is ``DEFAULT_SAFETY_RATIO``.
"""

import math
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from training_data_packer.metadata import (
    Metadata,
    get_in_suffix,
    get_matching_part,
    get_megatron_lm_tokenizers,
    get_source_dir,
    read_metadata,
)
from training_data_packer.mode.merge import is_single_task_merge
from training_data_packer.utils.file import find_files

GIB = 1024**3
DEFAULT_SAFETY_RATIO = 3.0
GIB_PER_MINUTE = 1.0
IDEAL_TASK_MINUTES = 5 * 60
PACKAGE_PARALLEL_FILES = 8
PARTITION_MAX_MINUTES = {"small": 72 * 60, "standard": 48 * 60}
PARTITION_MAX_TASKS = {"small": 200, "standard": 100}


@dataclass(frozen=True, slots=True)
class StepPlan:
    """Array size and time limit for one SLURM step."""

    step: str
    partition: str
    array_size: int
    minutes: float

    def time_limit_hours(self) -> int:
        """Return the estimated time rounded up to whole hours, at least one hour."""
        return max(1, math.ceil(self.minutes / 60))

    def slurm_args(self) -> str:
        """Return the step as ``sbatch`` arguments."""
        return f"--partition={self.partition} --array=0-{self.array_size - 1} --time={self.time_limit_hours()}:00:00"


def plan_step(
    step: str,
    partition: str,
    unit_sizes: list[int],
    parallel: int,
    max_tasks: int,
    safety_ratio: float = 1.0,
) -> StepPlan:
    """Calculate the array size and the time needed per task for a step.

    The array size is chosen so a task ideally runs for ``IDEAL_TASK_MINUTES``. A task never runs shorter than the
    time needed for the largest unit, since a unit can not be split between tasks. The array size is limited by
    ``max_tasks``, the number of units and the maximum number of tasks of the partition. All times include the
    safety ratio.

        :param step: name of the step
        :param partition: SLURM partition the step runs in
        :param unit_sizes: sizes in bytes of the units, e.g. files, that can not be split between tasks
        :param parallel: number of units a task processes in parallel
        :param max_tasks: maximum number of tasks the step can use
        :param safety_ratio: multiplier of the estimated time, e.g. 1.5 reserves 50% more time
        :returns: the plan for the step
        :raises ValueError: if a task needs more time than the partition allows
    """
    total_minutes = sum(unit_sizes) / GIB / GIB_PER_MINUTE * safety_ratio
    largest_minutes = max(unit_sizes, default=0) / GIB / GIB_PER_MINUTE * safety_ratio
    max_tasks = max(1, min(max_tasks, len(unit_sizes), PARTITION_MAX_TASKS[partition]))

    array_size = min(max(1, math.ceil(total_minutes / (parallel * IDEAL_TASK_MINUTES))), max_tasks)
    units_per_task = math.ceil(len(unit_sizes) / array_size)
    minutes = max(largest_minutes, total_minutes / (array_size * max(1, min(parallel, units_per_task))))

    plan = StepPlan(step, partition, array_size, minutes)
    if plan.time_limit_hours() * 60 > PARTITION_MAX_MINUTES[partition]:
        raise ValueError(
            f"Step {step} needs {plan.time_limit_hours()} hours per task with {array_size} tasks, "
            f"partition {partition} allows {PARTITION_MAX_MINUTES[partition] // 60} hours",
        )
    logger.info(
        f"Step {step}: {len(unit_sizes)} units, {sum(unit_sizes) / GIB:.1f} GiB total, "
        f"{max(unit_sizes, default=0) / GIB:.1f} GiB largest, safety ratio {safety_ratio}, "
        f"{array_size} tasks of {minutes:.0f} minutes",
    )
    return plan


def _get_part_file_sizes(metadata: Metadata) -> dict[str, list[int]]:
    source_dir = get_source_dir(metadata)
    part_sizes: dict[str, list[int]] = {}
    for file in find_files(source_dir, get_in_suffix(metadata, "release")):
        _, part = get_matching_part(metadata, file, section_name="release")
        if part is not None:
            part_sizes.setdefault(part, []).append(file.stat().st_size)
    return part_sizes


def plan_pipeline(metadata: Metadata, safety_ratio: float = DEFAULT_SAFETY_RATIO) -> list[StepPlan]:
    """Plan the package, merge and tokenize steps of the pipeline.

    :param metadata: metadata of the collection, read in release mode
    :param safety_ratio: multiplier of the estimated time, e.g. 1.5 reserves 50% more time
    :returns: one plan per step, in pipeline order
    :raises ValueError: if no files are included in a release part, the safety ratio is not positive, or a step does
        not fit in its partition
    """
    if safety_ratio < 1:
        raise ValueError(f"Safety ratio must be positive greater than or equal to 1, got {safety_ratio}")
    part_sizes = _get_part_file_sizes(metadata)
    file_sizes = [size for sizes in part_sizes.values() for size in sizes]
    if len(file_sizes) == 0:
        logger.error("No files included in release parts, probably error in metadata.yaml")
        raise ValueError("No files detected")

    package = plan_step("package", "small", file_sizes, PACKAGE_PARALLEL_FILES, len(file_sizes), safety_ratio)

    if is_single_task_merge(metadata, sorted(part_sizes)):
        merge = plan_step("merge", "small", [sum(file_sizes)], 1, 1, safety_ratio)
    else:
        merge_sizes = [sum(sizes) for sizes in part_sizes.values()]
        merge = plan_step("merge", "small", merge_sizes, 1, len(merge_sizes), safety_ratio)

    # The merged files do not exist yet, use the source files as estimate of the files to tokenize.
    tokenizer_count = max(1, len(get_megatron_lm_tokenizers(metadata)))
    tokenize_sizes = file_sizes * tokenizer_count
    tokenize = plan_step("tokenize", "standard", tokenize_sizes, 1, len(tokenize_sizes), safety_ratio)

    return [package, merge, tokenize]


def process(collection_dir: Path, safety_ratio: float = DEFAULT_SAFETY_RATIO) -> list[StepPlan]:
    """Plan the pipeline for a collection.

    :param collection_dir: collection directory containing ``metadata.yaml``
    :param safety_ratio: multiplier of the estimated time, e.g. 1.5 reserves 50% more time
    :returns: one plan per step, in pipeline order
    """
    metadata = read_metadata(collection_dir.joinpath("metadata.yaml"), log_content=False)
    metadata["_internal"]["mode"] = "release"
    return plan_pipeline(metadata, safety_ratio)
