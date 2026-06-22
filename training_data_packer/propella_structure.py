import argparse
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

from training_data_packer.processor.propella import SourceToPropellaMapper
from training_data_packer.storage.propella import get_lookup_fn
from training_data_packer.utils import metrics
from training_data_packer.utils.file import GenericJsonlReader, JsonlZstWriter, change_suffix, find_files
from training_data_packer.utils.metadata import get_metadata_value, read_metadata
from training_data_packer.utils.slurm import get_my_slurm_tasks


def process(collection_dir: Path, propella_dir: Path, part: str = "", slurm: bool = False) -> None:
    """
    Process all files in collection_dir/source.
    For each record, if its ID exists in propella_dir parquet files, write it to the
    output within the collection_dir.
    """
    output_dir = collection_dir.joinpath("propella-4b").joinpath(part)
    source_dir = collection_dir.joinpath("source").joinpath(part)
    metadata = read_metadata(collection_dir.joinpath("metadata.yaml"))

    suffix = get_metadata_value(metadata, "source.default.suffix", metadata["suffix"])
    all_files = find_files(source_dir, suffix)
    if slurm:
        task_files = get_my_slurm_tasks(all_files)
    else:
        logger.info("Not a SLURM task, processing all files")
        task_files = all_files
    if propella_dir.is_file():
        output_dir = output_dir.joinpath(propella_dir.stem)
    propella_lookup_fn, propella_metrics = get_lookup_fn(propella_dir, "id")
    for source_file in task_files:
        rel_path = source_file.relative_to(source_dir)
        output_file = _compute_output_filename(output_dir, rel_path, metadata)
        if output_file.exists():
            logger.info(f"Skipping {source_file}, output already exists")
        else:
            process_file(metadata, propella_lookup_fn, source_file, output_file, propella_metrics)


def _compute_output_filename(output_dir: Path, rel_path: Path, metadata: dict[str, Any]) -> Path:
    output_file = change_suffix(
        output_dir.joinpath(rel_path),
        metadata["suffix"],
        ".jsonl.zst",
    )
    return output_file


def process_file(
    metadata: dict[str, Any],
    propella_lookup_fn: Callable,
    source_file: Path,
    output_file: Path,
    global_metrics: dict[str, Any],
):
    tmp_output_file = output_file.parent.joinpath("." + output_file.name)
    os.makedirs(output_file.parent, exist_ok=True)

    source_reader = GenericJsonlReader(source_file)
    source_to_propella_mapper = SourceToPropellaMapper(metadata, propella_lookup_fn)
    mapped_iter = map(source_to_propella_mapper.get_mapper(), source_reader.read())
    writer = JsonlZstWriter(tmp_output_file)
    writer.write(mapped_iter)

    os.rename(tmp_output_file, output_file)

    metrics_collection = metrics.collect_metrics(
        source_reader,
        source_to_propella_mapper,
        writer,
        global_metrics,
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
    parser.add_argument("--part", help="Part to process, eg deu_Latn", default="")
    parser.add_argument(
        "-s",
        "--slurm",
        help="Only process files for my slurm partition",
        action="store_true",
    )
    args = parser.parse_args()

    process(Path(args.collection_dir), Path(args.propella), args.part, args.slurm)


if __name__ == "__main__":
    main()
