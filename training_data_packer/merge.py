import argparse
import io
import os
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import zstandard as zstd
from loguru import logger

from training_data_packer.utils.file import find_files
from training_data_packer.utils.metadata import (
    get_all_part_names,
    get_matching_part,
    get_shard_size_documents,
    read_metadata,
)
from training_data_packer.utils.slurm import get_my_slurm_tasks


def merge(input_files: Iterable[Path], destination_dir: Path, docs_per_shard: int, file_prefix: str):
    logger.info(f"Writing to directory {destination_dir}, using prefix {file_prefix}")
    os.makedirs(destination_dir, exist_ok=True)

    dctx = zstd.ZstdDecompressor()
    cctx = zstd.ZstdCompressor(level=3)

    file_idx = 0
    docs_written = 0
    out_f = None
    writer = None

    try:
        for file_path in input_files:
            with open(file_path, "rb") as in_f:
                with dctx.stream_reader(in_f) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                    for line in text_stream:
                        if out_f is None or (docs_written >= docs_per_shard) or docs_written == 0:
                            if out_f:
                                writer.close()
                                out_f.close()

                            output_path = destination_dir.joinpath(f"{file_prefix}_{file_idx:04d}.jsonl.zst")
                            out_f = open(output_path, "wb")
                            writer = cctx.stream_writer(out_f)

                            docs_written = 0
                            file_idx += 1
                        writer.write(line.encode("utf-8"))
                        docs_written += 1
    finally:
        if writer:
            writer.close()
        if out_f:
            out_f.close()


def process(collection_dir: Path, workers: int = 1, slurm: bool = False):
    metadata = read_metadata(collection_dir.joinpath("metadata.yaml"))
    input_dir = collection_dir.joinpath("release-raw")
    output_dir = collection_dir.joinpath("release")

    parts = get_all_part_names(metadata)
    logger.info(f"Found {len(parts)} parts")

    part_config, _ = get_matching_part(metadata, parts[0])
    if "pack" in part_config and part_config["pack"] == "flat" and "prefix" not in part_config:
        # This config requires single threaded
        workers = 1
        parts = ["default"]
        logger.info("All parts will be flatten into one. This will run single threaded.")

    if slurm:
        task_parts = get_my_slurm_tasks(parts)
    else:
        logger.info("Not a SLURM task, processing all files")
        task_parts = parts

    if workers > 1:
        jobs = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for part_name in task_parts:
                part_config, _ = get_matching_part(metadata, part_name)
                logger.info(f"Processing part {part_name} with config {part_config}")
                flat_output = part_config["pack"] == "flat"
                metadata["suffix"] = ".jsonl.zst"
                files = find_files(input_dir.joinpath(part_name), metadata["suffix"])
                logger.info(f"Processing part {part_name} with {len(files)} files")
                docs_per_shard = get_shard_size_documents(part_config)
                job = executor.submit(
                    merge,
                    files,
                    output_dir if flat_output else output_dir.joinpath(part_name),
                    docs_per_shard,
                    part_config.get("prefix", "shard"),
                )
                jobs.append(job)
            executor.shutdown()
            for n, job in enumerate(jobs):
                if job.exception() is not None:
                    logger.error(f"There were an exception thrown for release {task_parts[n]}: {job.exception()}")
    else:
        if parts == ["default"]:
            metadata["suffix"] = ".jsonl.zst"
            files = find_files(input_dir, metadata["suffix"])
            docs_per_shard = get_shard_size_documents(part_config)
            merge(
                files,
                output_dir,
                docs_per_shard,
                "shard",
            )
        else:
            for part_name in task_parts:
                part_config, _ = get_matching_part(metadata, part_name)
                logger.info(f"Processing part {part_name} with config {part_config}")
                flat_output = part_config["pack"] == "flat"
                metadata["suffix"] = ".jsonl.zst"
                files = find_files(input_dir.joinpath(part_name), metadata["suffix"])
                logger.info(f"Processing part {part_name} with {len(files)} files")
                docs_per_shard = get_shard_size_documents(part_config)
                merge(
                    files,
                    output_dir if flat_output else output_dir.joinpath(part_name),
                    docs_per_shard,
                    part_config.get("prefix", "shard"),
                )


def main():
    parser = argparse.ArgumentParser(
        prog="training-data-packer",
        description="Pack training data from input directory to output directory.",
    )
    parser.add_argument("--collection-dir", help="Collection directory containing data", required=True)
    parser.add_argument("-w", "--workers", help="Number of workers, default is 1", type=int, default=1)
    parser.add_argument(
        "-s",
        "--slurm",
        help="Only process files for my slurm partition",
        action="store_true",
    )
    args = parser.parse_args()
    process(
        Path(args.collection_dir),
        args.workers,
        args.slurm,
    )


if __name__ == "__main__":
    main()
