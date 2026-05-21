import argparse
import json
from pathlib import Path

from loguru import logger

from training_data_packer.utils import metrics


def collect_metrics(work_dir: Path):
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

    summary = metrics.aggregate_metrics(all_metrics)

    output_file = work_dir.joinpath("metrics.json")
    metrics.write_metrics_to_file(summary, output_file)
    logger.info(f"Summary written to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        prog="oellm-collect-metrics",
        description="Collect and summarize metrics from a collection directory.",
    )
    parser.add_argument("--collection-dir", help="Collection directory containing data", required=True)
    args = parser.parse_args()

    collect_metrics(Path(args.collection_dir).joinpath("release_raw"))


if __name__ == "__main__":
    main()
