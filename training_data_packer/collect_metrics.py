import argparse
import json
from pathlib import Path

from loguru import logger

from training_data_packer.utils import metrics
from training_data_packer.utils.metadata import get_all_part_names, read_metadata


def collect_metrics_for_dir(work_dir: Path, output_file: Path):
    logger.info(f"Collecting metrics from {work_dir}")

    metrics_files = list(work_dir.rglob(".*.metrics.json"))
    logger.info(f"Found {len(metrics_files)} metrics files")

    all_metrics = []
    for metrics_file in metrics_files:
        try:
            with open(metrics_file) as f:
                all_metrics.append(json.load(f))
        except Exception as e:
            logger.error(f"Failed to read {metrics_file}: {e}")

    if not all_metrics:
        logger.warning(f"No metrics files found in {work_dir}")
        return

    summary = metrics.aggregate_metrics(all_metrics)
    metrics.write_metrics_to_file(summary, output_file)
    logger.info(f"Summary written to {output_file}")


def collect_metrics(work_dir: Path):
    collect_metrics_for_dir(work_dir, work_dir.joinpath("metrics.json"))


def main():
    parser = argparse.ArgumentParser(
        prog="oellm-collect-metrics",
        description="Collect and summarize metrics from a collection directory.",
    )
    parser.add_argument("--collection-dir", help="Collection directory containing data", required=True)
    args = parser.parse_args()

    collection_dir = Path(args.collection_dir)
    metadata_path = collection_dir.joinpath("metadata.yaml")
    release_raw_dir = collection_dir.joinpath("release_raw")

    if metadata_path.exists():
        metadata = read_metadata(metadata_path)
        if metadata.get("release", {}).get("default", {}).get("pack") == "tree":
            parts = get_all_part_names(metadata)
            for part in parts:
                part_dir = release_raw_dir.joinpath(part)
                collect_metrics(part_dir)

    # Always collect metrics global.
    collect_metrics(release_raw_dir)


if __name__ == "__main__":
    main()
