import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from loguru import logger

from training_data_packer.mode.release import package_file
from training_data_packer.mode.sample import sample_file
from training_data_packer.utils.file import change_suffix, find_files
from training_data_packer.utils.metadata import get_metadata_value, read_metadata
from training_data_packer.utils.slurm import get_my_slurm_tasks


def process(
    collection_dir: Path, workers=1, slurm: bool = False, part: str | None = None, mode: str = "release"
) -> None:
    metadata = read_metadata(collection_dir.joinpath("metadata.yaml"))
    source_dir = collection_dir.joinpath(metadata["release"]["default"]["input"])
    contamination_dir = collection_dir.joinpath("nemo-curator")
    pii_dir = collection_dir.joinpath("openai-privacy-filter")
    propella_dir = collection_dir.joinpath("propella-4b")
    match mode:
        case "release":
            output_dir = collection_dir.joinpath("release-raw")
        case "sample":
            output_dir = collection_dir.joinpath("sample")
        case _:
            raise ValueError(f"Undefined mode {mode}")

    suffix = get_metadata_value(metadata, "source.default.suffix", metadata["suffix"])
    all_files = find_files(source_dir, suffix, part)
    logger.info(f"Found {len(all_files)} files")

    if slurm:
        task_files = get_my_slurm_tasks(all_files)
    else:
        logger.info("Not a SLURM task, processing all files")
        task_files = all_files

    if workers > 1:
        jobs = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for src_file in task_files:
                contamination_file, pii_file, propella_file, out_file = _calculate_file_paths(
                    src_file, source_dir, contamination_dir, pii_dir, propella_dir, output_dir, metadata
                )
                match mode:
                    case "release":
                        job = executor.submit(
                            package_file,
                            src_file,
                            metadata,
                            contamination_file,
                            pii_file,
                            propella_file,
                            out_file,
                        )
                    case "sample":
                        job = executor.submit(sample_file, src_file, metadata, propella_file, out_file)
                    case _:
                        raise ValueError(f"Undefined mode {mode}")
                jobs.append(job)
            executor.shutdown()
            for n, job in enumerate(jobs):
                if job.exception() is not None:
                    logger.error(f"There were an exception thrown for file {task_files[n]}: {job.exception()}")
    else:
        for src_file in task_files:
            logger.debug(f"Processing file {src_file}")
            contamination_file, pii_file, propella_file, out_file = _calculate_file_paths(
                src_file, source_dir, contamination_dir, pii_dir, propella_dir, output_dir, metadata
            )
            match mode:
                case "release":
                    package_file(src_file, metadata, contamination_file, pii_file, propella_file, out_file)
                case "sample":
                    sample_file(src_file, metadata, propella_file, out_file)
                case _:
                    raise ValueError(f"Undefined mode {mode}")


def _calculate_file_paths(
    src_file: Path,
    source_dir: Path,
    contamination_dir: Path,
    pii_dir: Path,
    propella_dir: Path,
    output_dir: Path,
    metadata: dict,
) -> tuple[Path, Path, Path, Path]:
    rel_file_path = Path(str(src_file)[len(str(source_dir)) + 1 :])

    contamination_suffix = get_metadata_value(metadata, "annotations.contamination.suffix", default=metadata["suffix"])
    contamination_file = change_suffix(
        contamination_dir.joinpath(rel_file_path), metadata["suffix"], contamination_suffix
    )

    pii_suffix = get_metadata_value(metadata, "annotations.pii.suffix", default=metadata["suffix"])
    pii_file = change_suffix(pii_dir.joinpath(rel_file_path), metadata["suffix"], pii_suffix)

    propella_suffix = get_metadata_value(metadata, "annotations.pii.propella-4b", default=metadata["suffix"])
    propella_file = change_suffix(propella_dir.joinpath(rel_file_path), metadata["suffix"], propella_suffix)

    out_file = output_dir.joinpath(change_suffix(rel_file_path, metadata["suffix"], ".jsonl.zst"))
    return contamination_file, pii_file, propella_file, out_file


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
