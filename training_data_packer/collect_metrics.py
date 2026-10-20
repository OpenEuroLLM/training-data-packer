import argparse
from pathlib import Path

from training_data_packer.metadata import read_metadata
from training_data_packer.utils.metrics import merge_metrics


def main():
    parser = argparse.ArgumentParser(
        prog="oellm-collect-metrics",
        description="Collect and summarize metrics from a collection directory.",
    )
    parser.add_argument("--collection-dir", help="Collection directory containing data", required=True)
    args = parser.parse_args()

    collection_dir = Path(args.collection_dir)
    metadata_path = collection_dir.joinpath("metadata.yaml")
    release_raw_dir = collection_dir.joinpath("release-raw")

    if metadata_path.exists():
        metadata = read_metadata(metadata_path)
        if metadata.get("release", {}).get("default", {}).get("pack") == "tree":
            part_names = metadata.get_all_part_names("release")
            for part in part_names:
                part_dir = release_raw_dir.joinpath(part)
                merge_metrics(part_dir)

    # Always collect metrics global.
    merge_metrics(release_raw_dir)


if __name__ == "__main__":
    main()
