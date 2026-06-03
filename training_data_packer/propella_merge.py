import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from loguru import logger

from training_data_packer.processor.propella import MergePropellaRecords
from training_data_packer.utils import metrics
from training_data_packer.utils.file import (
    GenericJsonlReader,
    JsonlZstWriter,
    change_suffix,
    find_files,
    get_subdirectories,
)
from training_data_packer.utils.metadata import get_metadata_value, read_metadata
from training_data_packer.utils.metrics import read_metrics_from_file
from training_data_packer.utils.slurm import get_my_slurm_tasks


def process(collection_dir: Path, part: str = "", workers=1, slurm: bool = False) -> None:
    output_dir = collection_dir.joinpath("propella-4b").joinpath(part)
    source_dir = collection_dir.joinpath("source").joinpath(part)
    metadata = read_metadata(collection_dir.joinpath("metadata.yaml"))

    # Find all source file and take their names
    all_names = list(map(lambda x: x.name, find_files(source_dir, metadata)))

    for f in all_names:
        process_file(metadata, f, output_dir)
    if slurm:
        task_names = get_my_slurm_tasks(all_names)
    else:
        logger.info("Not a SLURM task, processing all files")
        task_names = all_names

    if workers > 1:
        jobs = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for src_name in task_names:
                job = executor.submit(process_file, metadata, src_name, output_dir)
                jobs.append(job)
            executor.shutdown()
            for n, job in enumerate(jobs):
                if job.exception() is not None:
                    logger.error(f"There were an exception thrown for file {task_names[n]}: {job.exception()}")
    else:
        for src_name in task_names:
            logger.debug(f"Processing {src_name}")
            process_file(metadata, src_name, output_dir)


def process_file(metadata: dict[str, Any], source_name: str, propella_dir: Path):
    """
    Processes and merges JSONL files from subdirectories within the specified
    propella directory. The function constructs the output path and checks for
    existence to avoid redundant processing. It establishes a list of read
    iterators for the source file found in every subdirectory, merges the
    corresponding records using an ID extracted from the provided metadata, and
    writes the compressed result atomically via a temporary file. Furthermore,
    it collects metric data from each subdirectory, aggregates them, and persists
    the collected metrics to a file.

    :param metadata: Dictionary containing configuration data used to extract
                     the necessary ID for merging records.
    :param source_name: The name of the file to be processed across subdirectories
                        and used for the final output destination.
    :param propella_dir: The directory path containing the subdirectories to
                         search for source files and the target location for the
                         merged output.
    :return: None
    """
    new_name = change_suffix(
        source_name,
        metadata["suffix"],
        get_metadata_value(metadata, "annotations.propella-4b.suffix", metadata["suffix"]),
    )
    out_file_name = propella_dir.joinpath(new_name)

    if out_file_name.exists():
        logger.info(f"File {out_file_name} already exist, skipping")
        return
    tmp_output_file = propella_dir.joinpath("." + source_name)
    read_iterators = []
    directories_to_process = get_subdirectories(propella_dir)
    for d in directories_to_process:
        read_iterators.append(GenericJsonlReader(d.joinpath(source_name)).read())
    zip_iter = zip(*read_iterators, strict=True)
    map_processor = MergePropellaRecords(metadata["id"])
    map_iter = map(map_processor.get_mapper(), zip_iter)

    writer = JsonlZstWriter(tmp_output_file)
    writer.write(map_iter)

    os.rename(tmp_output_file, out_file_name)

    metrics_from_dir = []
    for d in directories_to_process:
        m = read_metrics_from_file(d.joinpath(f".{source_name}.metrics.json"))
        metrics_from_dir.append({d.name: m})
    metrics_collection = metrics.collect_metrics(map_processor, {"propella_sub": metrics_from_dir})
    metrics_filename = propella_dir.joinpath("." + source_name + ".metrics.json")
    metrics.write_metrics_to_file(metrics_collection, metrics_filename)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="oellm-propella-merge",
        description="When processing propella directories too large for memory one file at a time must be processed. "
        "This merges multiple propella files into one.",
    )
    parser.add_argument("--collection-dir", help="Directory containing source jsonl.zst files", required=True)
    parser.add_argument("--part", help="Part to process, eg deu_Latn", default="")
    parser.add_argument("-w", "--workers", help="Number of workers, default is 1", type=int, default=1)
    parser.add_argument(
        "-s",
        "--slurm",
        help="Only process files for my slurm partition",
        action="store_true",
    )
    args = parser.parse_args()

    process(Path(args.collection_dir), args.part, args.workers, args.slurm)


if __name__ == "__main__":
    main()
