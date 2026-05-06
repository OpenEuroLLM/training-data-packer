import argparse
import io
import os
import sys
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import orjson as json
import zstandard as zstd
from loguru import logger

from training_data_packer.utils.file import find_jsonl_zst_files, get_directories_n_levels_down
from training_data_packer.utils.metadata import read_metadata
from training_data_packer.utils.slurm import get_my_slurm_tasks


def merge(input_files: Iterable[Path], destination_dir: Path, token_size: float, shard_size: int, file_prefix: str):
    os.makedirs(destination_dir, exist_ok=True)

    dctx = zstd.ZstdDecompressor()
    cctx = zstd.ZstdCompressor(level=3)

    file_idx = 0
    current_token_sum = 0

    # Initialize the first output file
    out_f = None
    writer = None

    try:
        for file_path in input_files:
            with open(file_path, "rb") as in_f:
                with dctx.stream_reader(in_f) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                    for line in text_stream:
                        data = json.loads(line)
                        text_tokens = len(data["text"]) / token_size

                        if out_f is None or (current_token_sum + text_tokens > shard_size) or current_token_sum == 0:
                            if out_f:
                                writer.close()
                                out_f.close()

                            output_path = destination_dir.joinpath(f"{file_prefix}_{file_idx:04d}.jsonl.zst")
                            out_f = open(output_path, "wb")
                            writer = cctx.stream_writer(out_f)

                            current_token_sum = 0
                            file_idx += 1

                        encoded_line = json.dumps(data) + b"\n"
                        writer.write(encoded_line)
                        current_token_sum += text_tokens
    finally:
        if writer:
            writer.close()
        if out_f:
            out_f.close()


def process(collection_dir: Path, token_size: float, shard_size: int, workers=1, slurm=False):
    metadata = read_metadata(collection_dir.joinpath("metadata.yaml"))
    input_dir = collection_dir.joinpath("release_raw")
    output_dir = collection_dir.joinpath("release")

    if output_dir.exists():
        logger.info(f"Out data directory {output_dir} already exist, exiting")
        sys.exit(1)

    parts = sorted(metadata["release"].keys())
    logger.info(f"Found {len(parts)} parts")

    if slurm:
        task_parts = get_my_slurm_tasks(parts)
    else:
        logger.info("Not a SLURM task, processing all files")
        task_parts = parts

    if workers > 1:
        jobs = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for part_name in task_parts:
                part_config = metadata["release"][part_name]
                flat_output = part_config["pack"] == "flat"
                files = find_jsonl_zst_files(input_dir.joinpath(part_name))
                job = executor.submit(
                    merge,
                    files,
                    output_dir if flat_output else output_dir.joinpath(part_name),
                    token_size,
                    shard_size,
                    part_config["prefix"]
                )
                jobs.append(job)
            executor.shutdown()
            for n, job in enumerate(jobs):
                if job.exception() is not None:
                    logger.error(f"There were an exception thrown for release {task_parts[n]}: {job.exception()}")
    else:
        for part_name in task_parts:
            part_config = metadata["release"][part_name]
            flat_output = part_config["pack"] == "flat"
            files = find_jsonl_zst_files(input_dir.joinpath(part_name))
            merge(
                files,
                output_dir if flat_output else output_dir.joinpath(part_name),
                token_size,
                shard_size,
                part_config["prefix"]
            )



def main():
    parser = argparse.ArgumentParser(
        prog="training-data-packer",
        description="Pack training data from input directory to output directory.",
    )
    parser.add_argument("--collection-dir", help="Collection directory containing data", required=True)
    parser.add_argument("-w", "--workers", help="Number of workers, default is 1", type=int, default=1)
    parser.add_argument("-t", "--token-size", help="Characters per token", type=float, default=4.25)
    parser.add_argument("--shard-size", help="Tokens per shard", type=int, default=100_000_000_000_000)
    parser.add_argument(
        "-s",
        "--slurm",
        help="Only process files for my slurm partition",
        action="store_true",
    )
    args = parser.parse_args()
    process(
        Path(args.collection_dir),
        args.token_size,
        args.shard_size,
    )


if __name__ == "__main__":
    main()
