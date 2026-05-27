import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from loguru import logger

from training_data_packer.storage.propella import get_lookup_fn
from training_data_packer.utils import metrics
from training_data_packer.utils.file import GenericJsonlReader, JsonlZstWriter, find_files
from training_data_packer.utils.metadata import read_metadata
from training_data_packer.utils.slurm import get_my_slurm_tasks

from .processor.propella import SourceToPropellaMapper


def process(collection_dir: Path, propella_dir: Path, workers=1, slurm: bool = False) -> None:
    """
    Process all files in collection_dir/source.
    For each record, if its ID exists in propella_dir parquet files, write it to the
    output within the collection_dir.
    """
    output_dir = collection_dir.joinpath("propella")
    source_dir = collection_dir.joinpath("source")
    metadata = read_metadata(collection_dir.joinpath("metadata.yaml"))

    all_files = find_files(source_dir, metadata)
    if slurm:
        task_files = get_my_slurm_tasks(all_files)
    else:
        logger.info("Not a SLURM task, processing all files")
        task_files = all_files

    if workers > 1:
        jobs = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for source_file in task_files:
                rel_path = source_file.relative_to(source_dir)
                output_file = output_dir.joinpath(rel_path)

                if output_file.exists():
                    logger.info(f"Skipping {source_file}, output already exists")
                else:
                    job = executor.submit(
                        process_file,
                        metadata,
                        propella_dir,
                        source_file,
                        output_file,
                    )
            executor.shutdown()
            for n, job in enumerate(jobs):
                if job.exception() is not None:
                    logger.error(f"There were an exception thrown for file {task_files[n]}: {job.exception()}")
    else:
        for source_file in task_files:
            rel_path = source_file.relative_to(source_dir)
            output_file = output_dir.joinpath(rel_path)

            if output_file.exists():
                logger.info(f"Skipping {source_file}, output already exists")
            else:
                process_file(metadata, propella_dir, source_file, output_file)


def process_file(metadata: dict[str, Any], propella_dir: Path, source_file: Path, output_file: Path):
    tmp_output_file = output_file.parent.joinpath("." + output_file.name)
    os.makedirs(output_file.parent, exist_ok=True)

    source_reader = GenericJsonlReader(source_file)
    propella_lookup_fn = get_lookup_fn(propella_dir, metadata["id"])
    source_to_propella_mapper = SourceToPropellaMapper(metadata["id"], propella_lookup_fn)
    mapped_iter = map(source_to_propella_mapper.get_mapper(), source_reader.read())
    writer = JsonlZstWriter(tmp_output_file)
    writer.write(mapped_iter)

    os.rename(tmp_output_file, output_file)

    metrics_collection = metrics.collect_metrics(
        source_reader,
        source_to_propella_mapper,
        writer,
    )
    metrics_filename = output_file.parent.joinpath("." + output_file.name + ".metrics.json")
    metrics.write_metrics_to_file(metrics_collection, metrics_filename)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="oellm-propella-structure",
        description="Structure data based on propella parquet records.",
    )
    parser.add_argument("--collection-dir", help="Directory containing source jsonl.zst files", required=True)
    parser.add_argument("--propella", help="Directory containing propella parquet files", required=True)
    parser.add_argument("-w", "--workers", help="Number of workers, default is 1", type=int, default=1)
    parser.add_argument(
        "-s",
        "--slurm",
        help="Only process files for my slurm partition",
        action="store_true",
    )
    args = parser.parse_args()

    process(Path(args.collection_dir), Path(args.propella), args.workers, args.slurm)


if __name__ == "__main__":
    main()
