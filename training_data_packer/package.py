import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from loguru import logger

from training_data_packer.mode.release import package_file
from training_data_packer.mode.sample import sample_file
from training_data_packer.utils.file import find_files
from training_data_packer.utils.metadata import calculate_file_path, get_in_suffix, get_source_dir, read_metadata
from training_data_packer.utils.slurm import get_my_slurm_tasks


def process(
    collection_dir: Path, workers=1, slurm: bool = False, part: str | None = None, mode: str = "release"
) -> None:
    metadata = read_metadata(collection_dir.joinpath("metadata.yaml"))
    metadata["_internal"]["mode"] = mode
    source_dir = get_source_dir(metadata)
    contamination_dir = collection_dir.joinpath("nemo-curator")
    pii_dir = collection_dir.joinpath("openai-privacy-filter")
    propella_dir = collection_dir.joinpath("propella-4b")
    match mode:
        case "release":
            output_dir = collection_dir.joinpath("release-raw")
        case "sample":
            output_dir = collection_dir.joinpath(mode)
        case _:
            raise ValueError(f"Undefined mode {mode}")

    src_suffix = get_in_suffix(metadata, mode)
    all_files = find_files(source_dir, src_suffix, part)
    if len(all_files) == 0:
        logger.error("No files detected, probably error in metadata.yaml")
        raise ValueError("No files detected")
    logger.info(f"Found {len(all_files)} files")

    if slurm:
        task_files = get_my_slurm_tasks(all_files)
    else:
        logger.info("Not a SLURM task, processing all files")
        task_files = all_files

    if workers > 1:
        jobs = []
        fail = False
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for src_file in task_files:
                contamination_file = calculate_file_path(src_file, metadata, mode, contamination_dir)
                pii_file = calculate_file_path(src_file, metadata, mode, pii_dir)
                propella_file = calculate_file_path(src_file, metadata, mode, propella_dir)
                output_file = calculate_file_path(src_file, metadata, mode, output_dir)

                match mode:
                    case "release":
                        job = executor.submit(
                            package_file,
                            src_file,
                            metadata,
                            contamination_file,
                            pii_file,
                            propella_file,
                            output_file,
                        )
                    case "sample":
                        job = executor.submit(sample_file, src_file, metadata, propella_file, output_file)
                    case _:
                        raise ValueError(f"Undefined mode {mode}")
                jobs.append(job)
            executor.shutdown()
            for n, job in enumerate(jobs):
                if job.exception() is not None:
                    logger.error(f"There were an exception thrown for file {task_files[n]}: {job.exception()}")
                    fail = True
            if fail:
                raise RuntimeError("One or more workers failed")
    else:
        for src_file in task_files:
            logger.debug(f"Processing file {src_file}")
            contamination_file = calculate_file_path(src_file, metadata, mode, contamination_dir)
            pii_file = calculate_file_path(src_file, metadata, mode, pii_dir)
            propella_file = calculate_file_path(src_file, metadata, mode, propella_dir)
            output_file = calculate_file_path(src_file, metadata, mode, output_dir)
            match mode:
                case "release":
                    package_file(src_file, metadata, contamination_file, pii_file, propella_file, output_file)
                case "sample":
                    sample_file(src_file, metadata, propella_file, output_file)
                case _:
                    raise ValueError(f"Undefined mode {mode}")


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
    parser.add_argument("-p", "--part", help="Part to process, default is all")
    parser.add_argument(
        "-m", "--mode", help="Mode to run packager in", default="release", choices=["release", "sample"]
    )
    args = parser.parse_args()
    process(
        Path(args.collection_dir),
        workers=args.workers,
        slurm=args.slurm,
        part=args.part,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
