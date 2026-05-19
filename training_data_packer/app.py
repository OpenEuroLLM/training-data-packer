import argparse
import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from loguru import logger

from training_data_packer import sample_register
from training_data_packer.clean import AlignFieldNames, field_scrubber_factory
from training_data_packer.filters import filter_on_blocklist
from training_data_packer.jsonl_zst import JsonlZstWriter
from training_data_packer.pii_masking import PiiMasker
from training_data_packer.sampler import sampler_factory
from training_data_packer.utils.file import GenericJsonlReader, find_files
from training_data_packer.utils.metadata import get_matching_part, read_metadata
from training_data_packer.utils.slurm import get_my_slurm_tasks


def package_file(src_file: Path, metadata: dict, contamination_file: Path, pii_file: Path, out_file: Path):
    tmp_out_file = out_file.parent.joinpath("." + out_file.name)
    if out_file.exists():
        # File is already processed. Do not do it again
        logger.info(f"Skipping {out_file}, already exists")
        return
    if tmp_out_file.exists():
        logger.info(f"Remove old temporary file {tmp_out_file}")
        os.remove(tmp_out_file)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    pii_iter = AlignFieldNames(GenericJsonlReader(pii_file).read(), metadata, no_key_hierarchy=True)

    part_config, part_name = get_matching_part(metadata, src_file)

    src_reader = GenericJsonlReader(src_file)
    align_iter = AlignFieldNames(src_reader.read(), metadata)
    scrub_iter = field_scrubber_factory(align_iter, part_config)

    # After this comment are actual records removed. Processing cannot require zipping of dataset works.
    if "id" in metadata:
        pii_masked_iter = PiiMasker(scrub_iter, pii_iter)
        contamination_ids = [x["id"] for x in AlignFieldNames(GenericJsonlReader(contamination_file).read(), metadata)]
        logger.debug(f"Found {len(contamination_ids)} contamination ids for file {src_file}")
        filtered = filter_on_blocklist(pii_masked_iter, contamination_ids)
        if "block" in part_config:
            filtered = filter_on_blocklist(filtered, part_config["block"])
    else:
        logger.info("No id field in metadata, skipping pii, decontamination and blocklist")
        filtered = scrub_iter
    if "wds+register" in metadata:
        logger.info("WDS+register files")
        filtered = itertools.chain.from_iterable(map(sample_register.process_record, filtered))
    sampled = sampler_factory(filtered, metadata, src_file)

    JsonlZstWriter(tmp_out_file).write(sampled)
    os.rename(tmp_out_file, out_file)


def process(input_dir: Path, output_dir: Path, workers=1, slurm=False, release=None) -> None:
    metadata = read_metadata(input_dir.joinpath("metadata.yaml"))
    source_dir = input_dir.joinpath(metadata["release"]["default"]["input"])
    contamination_dir = input_dir.joinpath("contamination")
    pii_dir = input_dir.joinpath("pii")

    all_files = find_files(source_dir, metadata, release)
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
                contamination_file, pii_file, out_file = _calculate_file_paths(
                    src_file, source_dir, contamination_dir, pii_dir, output_dir, metadata
                )

                job = executor.submit(
                    package_file,
                    src_file,
                    metadata,
                    contamination_file,
                    pii_file,
                    out_file,
                )
                jobs.append(job)
            executor.shutdown()
            for n, job in enumerate(jobs):
                if job.exception() is not None:
                    logger.error(f"There were an exception thrown for file {task_files[n]}: {job.exception()}")
    else:
        for src_file in task_files:
            logger.debug(f"Processing file {src_file}")
            contamination_file, pii_file, out_file = _calculate_file_paths(
                src_file, source_dir, contamination_dir, pii_dir, output_dir, metadata
            )
            package_file(src_file, metadata, contamination_file, pii_file, out_file)


def _calculate_file_paths(
    src_file, source_dir: Path, contamination_dir: Path, pii_dir: Path, output_dir: Path, metadata: dict
) -> tuple[Path, Path, Path]:
    rel_file_path = Path(str(src_file)[len(str(source_dir)) + 1 :])
    contamination_file = contamination_dir.joinpath(rel_file_path)
    if not contamination_file.exists():
        contamination_file = contamination_dir.joinpath(rel_file_path.parent, rel_file_path.stem)
    pii_file = pii_dir.joinpath(rel_file_path)
    if "suffix" in metadata["annotations"]["pii"] and metadata["suffix"] != metadata["annotations"]["pii"]["suffix"]:
        pii_file = Path(str(pii_file).replace(metadata["suffix"], metadata["annotations"]["pii"]["suffix"]))
    if not pii_file.exists():
        pii_file = pii_dir.joinpath(rel_file_path.parent, rel_file_path.stem)
    out_file = output_dir.joinpath(rel_file_path.parent, rel_file_path.stem + ".zst")
    return contamination_file, pii_file, out_file


def main():
    parser = argparse.ArgumentParser(
        prog="training-data-packer",
        description="Pack training data from input directory to output directory.",
    )
    parser.add_argument("--input_dir", help="Input directory containing source data", required=True)
    parser.add_argument("--output_dir", help="Output directory for packed training data", required=True)
    parser.add_argument("-w", "--workers", help="Number of workers, default is 1", type=int, default=1)
    parser.add_argument(
        "-s",
        "--slurm",
        help="Only process files for my slurm partition",
        action="store_true",
    )
    parser.add_argument("-r", "--release", help="Release to process, default is all")
    args = parser.parse_args()
    process(
        Path(args.input_dir),
        Path(args.output_dir),
        workers=args.workers,
        slurm=args.slurm,
        release=args.release,
    )


if __name__ == "__main__":
    main()
