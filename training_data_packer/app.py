import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from loguru import logger

from training_data_packer.align import AlignFieldNames
from training_data_packer.decontaminate import Decontaminate
from training_data_packer.filters import filter_to_be_deleted
from training_data_packer.pii_masking import PiiMasker

from .jsonl_zst import JsonlZstReader, JsonlZstWriter


def find_jsonl_zst_files(input_dir: Path) -> list[Path]:
    return sorted(Path(input_dir).glob("[A-Za-z0-9]*.jsonl.zst"))


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

    contamination_iter = AlignFieldNames(JsonlZstReader(contamination_file).read(), metadata)
    pii_iter = AlignFieldNames(JsonlZstReader(pii_file).read(), metadata)

    src_reader = JsonlZstReader(src_file)
    align_iter = AlignFieldNames(src_reader.read(), metadata)
    decontaminated_iter = Decontaminate(align_iter, contamination_iter)
    pii_masked_iter = PiiMasker(decontaminated_iter, pii_iter)
    filtered = filter_to_be_deleted(pii_masked_iter)

    JsonlZstWriter(tmp_out_file).write(filtered)
    os.rename(tmp_out_file, out_file)

def main(input_dir: Path, output_dir: Path, workers=1) -> None:
    metadata = read_metadata(input_dir.joinpath("metadata.yaml"))
    source_dir = input_dir.joinpath("source")
    contamination_dir = input_dir.joinpath("contamination")
    pii_dir = input_dir.joinpath("pii")

    files = find_jsonl_zst_files(source_dir)
    logger.info(f"Found {len(files)} files")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for src_file in files:
            rel_file_path = str(src_file)[len(str(source_dir))+1:]
            contamination_file=os.path.join(contamination_dir, rel_file_path)
            pii_file=os.path.join(pii_dir, rel_file_path)
            out_file=output_dir.joinpath(rel_file_path)

            executor.submit(package_file, src_file, metadata, contamination_file, pii_file, out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="training-data-packer",
        description="Pack training data from input directory to output directory.",
    )
    parser.add_argument("--input_dir", help="Input directory containing source data")
    parser.add_argument("--output_dir", help="Output directory for packed training data")
    parser.add_argument("--workers", help="Number of workers, default is 1", type=int, default=1)
    args = parser.parse_args()
    main(Path(args.input_dir), Path(args.output_dir), workers=args.workers)
