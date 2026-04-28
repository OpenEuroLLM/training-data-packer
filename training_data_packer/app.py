import argparse
import os
from pathlib import Path

import yaml

from training_data_packer.align import AlignFieldNames
from training_data_packer.decontaminate import Decontaminate
from training_data_packer.filters import filter_to_be_deleted
from .jsonl_zst import JsonlZstReader, JsonlZstWriter


def find_jsonl_zst_files(input_dir: Path) -> list[Path]:
    return sorted(Path(input_dir).glob("[A-Za-z0-9]*.jsonl.zst"))


def read_metadata(file_path: Path) -> dict:
    with open(file_path, 'r') as file:
        metadata = yaml.safe_load(file)
        return metadata


def main(input_dir: Path, output_dir: Path) -> None:
    metadata = read_metadata(input_dir.joinpath("metadata.yaml"))
    source_dir = input_dir.joinpath("source")
    contamination_dir = input_dir.joinpath("contamination")
    pii_dir = input_dir.joinpath("pii")

    files = find_jsonl_zst_files(source_dir)
    print(f"Found {len(files)} files")

    for src_file in files:
        rel_file_path = str(src_file)[len(str(source_dir))+1:]
        contamination_file=os.path.join(contamination_dir, rel_file_path)
        pii_file=os.path.join(pii_dir, rel_file_path)
        out_file=os.path.join(output_dir, rel_file_path)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        src_reader = JsonlZstReader(src_file)
        align_iter = AlignFieldNames(src_reader.read(), metadata)
        decontaminated_iter = Decontaminate(align_iter, JsonlZstReader(contamination_file).read())
        filtered = filter_to_be_deleted(decontaminated_iter)
        JsonlZstWriter(out_file).write(filtered)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="training-data-packer",
        description="Pack training data from input directory to output directory.",
    )
    parser.add_argument("--input_dir", help="Input directory containing source data")
    parser.add_argument("--output_dir", help="Output directory for packed training data")
    args = parser.parse_args()
    main(Path(args.input_dir), Path(args.output_dir))
