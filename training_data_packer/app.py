import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from training_data_packer.align import AlignFieldNames
from training_data_packer.decontaminate import Decontaminate
from training_data_packer.filters import filter_on_blocklist, filter_to_be_deleted
from training_data_packer.pii_masking import PiiMasker
from training_data_packer.sampler import sampler_factory
from training_data_packer.utils.metadata import get_matching_release

from .jsonl_zst import JsonlZstReader, JsonlZstWriter


def find_jsonl_zst_files(source_dir: Path, release) -> list[Path]:
    if release is None:
        return sorted(Path(source_dir).glob("**/[A-Za-z0-9]*.jsonl.zst"))
    else:
        return sorted(Path(source_dir).glob(f"{release}/**/[A-Za-z0-9]*.jsonl.zst"))


def read_metadata(file_path: Path) -> dict:
    with open(file_path) as file:
        metadata = yaml.safe_load(file)
        return metadata

def package_file(src_file: Path, metadata: dict, contamination_file: str, pii_file: str, out_file: Path):
    tmp_out_file = out_file.parent.joinpath("." + out_file.name)
    if out_file.exists():
        # File is already processed. Do not do it again
        logger.info(f"Skipping {out_file}, already exists")
        return
    if tmp_out_file.exists():
        logger.info(f"Remove old temporary file {tmp_out_file}")
        os.remove(tmp_out_file)

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    release = get_matching_release(metadata, src_file)

    contamination_iter = AlignFieldNames(JsonlZstReader(contamination_file).read(), metadata, no_key_hierarchy = True)
    pii_iter = AlignFieldNames(JsonlZstReader(pii_file).read(), metadata, no_key_hierarchy = True)

    src_reader = JsonlZstReader(src_file)
    align_iter = AlignFieldNames(src_reader.read(), metadata)
    decontaminated_iter = Decontaminate(align_iter, contamination_iter)
    pii_masked_iter = PiiMasker(decontaminated_iter, pii_iter)

    # After this comment are actual records removed. Processing cannot require zipping of dataset works.

    sampled = sampler_factory(pii_masked_iter, metadata, src_file)
    filtered = filter_to_be_deleted(sampled)
    if "block" in release:
        filtered = filter_on_blocklist(filtered, release["block"])

    JsonlZstWriter(tmp_out_file).write(filtered)
    os.rename(tmp_out_file, out_file)


def extract_files_for_task(files:list[Any], task_count:int, task_id:int):
    size, rest = divmod(len(files), task_count)
    grouped_files =  [files[i * size + min(i, rest): (i + 1) * size + min(i + 1, rest)] for i in range(task_count)]
    return grouped_files[task_id-1]


def process(input_dir: Path, output_dir: Path, workers=1, slurm=False, release=None) -> None:
    metadata = read_metadata(input_dir.joinpath("metadata.yaml"))
    source_dir = input_dir.joinpath("source")
    contamination_dir = input_dir.joinpath("contamination")
    pii_dir = input_dir.joinpath("pii")

    all_files = find_jsonl_zst_files(source_dir, release)
    logger.info(f"Found {len(all_files)} files")

    if slurm:
        task_count = os.environ["SLURM_ARRAY_TASK_COUNT"]
        task_id = os.environ["SLURM_ARRAY_TASK_ID"]
        task_files = extract_files_for_task(all_files, int(task_count), int(task_id))
        logger.info(f"Slurm task id: {task_id} of {task_count}, processing {len(task_files)} files")
    else:
        logger.info("Not a SLURM task, processing all files")
        task_files = all_files

    if workers > 1:
        jobs = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for src_file in task_files:
                contamination_file, pii_file, out_file = _calculate_file_paths(src_file, source_dir, contamination_dir,
                                                                              pii_dir, output_dir)

                job = executor.submit(package_file, src_file, metadata, contamination_file, pii_file, out_file)
                jobs.append(job)
            executor.shutdown()
    else:
        for src_file in task_files:
            logger.debug(f"Processing file {src_file}")
            contamination_file, pii_file, out_file = _calculate_file_paths(src_file, source_dir, contamination_dir,
                                                                          pii_dir, output_dir)
            package_file(src_file, metadata, contamination_file, pii_file, out_file)


def _calculate_file_paths(src_file, source_dir: Path, contamination_dir: Path, pii_dir: Path, output_dir: Path) -> tuple[
    str, str, Path]:
    rel_file_path = str(src_file)[len(str(source_dir)) + 1:]
    contamination_file = os.path.join(contamination_dir, rel_file_path)
    pii_file = os.path.join(pii_dir, rel_file_path)
    out_file = output_dir.joinpath(rel_file_path)
    return contamination_file, pii_file, out_file


def main():
    parser = argparse.ArgumentParser(
        prog="training-data-packer",
        description="Pack training data from input directory to output directory.",
    )
    parser.add_argument("--input_dir", help="Input directory containing source data")
    parser.add_argument("--output_dir", help="Output directory for packed training data")
    parser.add_argument("-w", "--workers", help="Number of workers, default is 1", type=int, default=1)
    parser.add_argument("-s", "--slurm", help="Only process files for my slurm partition", action="store_true")
    parser.add_argument("-r", "--release", help="Release to process, default is all")
    args = parser.parse_args()
    process(Path(args.input_dir), Path(args.output_dir), workers=args.workers, slurm=args.slurm, release=args.release)

if __name__ == "__main__":
    main()
