import argparse
import os
from pathlib import Path

from loguru import logger

from training_data_packer.storage.propella import get_lookup_fn
from training_data_packer.utils import metrics
from training_data_packer.utils.file import GenericJsonlReader, JsonlZstWriter, find_files
from training_data_packer.utils.metadata import read_metadata

from .processor.propella import SourceToPropellaMapper


def process(collection_dir: Path, propella_dir: Path) -> None:
    """
    Process all files in collection_dir/source.
    For each record, if its ID exists in propella_dir parquet files, write it to the
    output within the collection_dir.
    """
    output_dir = collection_dir.joinpath("propella")
    source_dir = collection_dir.joinpath("source")
    metadata = read_metadata(collection_dir.joinpath("metadata.yaml"))

    all_files = find_files(source_dir, metadata)

    for source_file in all_files:
        rel_path = source_file.relative_to(source_dir)
        output_file = output_dir.joinpath(rel_path)

        if output_file.exists():
            logger.info(f"Skipping {source_file}, output already exists")
            continue

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
    args = parser.parse_args()

    process(Path(args.collection_dir), Path(args.propella))


if __name__ == "__main__":
    main()
