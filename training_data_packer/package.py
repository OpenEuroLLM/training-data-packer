import argparse
import sys
from pathlib import Path

from training_data_packer.mode import lint, merge, release, sample


def main():
    parser = argparse.ArgumentParser(
        prog="training-data-packer",
        description="Pack training data from input directory to output directory.",
    )
    parser.add_argument("--collection-dir", help="Directory for collection", required=True)
    parser.add_argument("-w", "--workers", help="Number of workers, default is 1", type=int, default=1)
    parser.add_argument(
        "-s",
        "--slurm",
        help="Only process files for my slurm partition",
        action="store_true",
    )
    parser.add_argument(
        "--token-count",
        help="Count tokens, slower processing",
        action="store_true",
    )
    parser.add_argument("-p", "--part", help="Part to process, default is all")
    parser.add_argument(
        "-m",
        "--mode",
        help="Mode to run packager in",
        default="release",
        choices=["lint", "release", "sample", "merge"],
    )
    args = parser.parse_args()

    match args.mode:
        case "lint":
            if args.slurm or args.workers > 1 or args.part is not None or args.token_count:
                raise ValueError("Lint mode does not support SLURM, multiple workers, part selection, or counting")
            if not lint.process(Path(args.collection_dir)):
                sys.exit(1)
        case "release":
            release.process(Path(args.collection_dir), args=vars(args))
        case "sample":
            sample.process(Path(args.collection_dir), args=vars(args))
        case "merge":
            if not merge.process(Path(args.collection_dir), args=vars(args)):
                sys.exit(2)
        case _:
            raise ValueError(f"Undefined mode {args.mode}. Use one of: lint, sample, release, or merge")


if __name__ == "__main__":
    main()
