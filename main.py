import argparse
from pathlib import Path

from training_data_packer.app import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="training-data-packer",
        description="Pack training data from input directory to output directory.",
    )
    parser.add_argument("--input_dir", help="Input directory containing source data")
    parser.add_argument("--output_dir", help="Output directory for packed training data")
    args = parser.parse_args()
    main(Path(args.input_dir), Path(args.output_dir))
