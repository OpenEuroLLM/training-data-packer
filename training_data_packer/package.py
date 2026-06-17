import argparse
import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from loguru import logger

from training_data_packer.processor.clean import AlignFieldNames, field_scrubber_factory
from training_data_packer.processor.filters import FilterOnBlocklist
from training_data_packer.processor.pii_masking import PIIMasker
from training_data_packer.processor.propella import propella_annotate_factory
from training_data_packer.processor.sample import sample_register
from training_data_packer.processor.sample.sampler import sampler_factory
from training_data_packer.utils import metrics
from training_data_packer.utils.file import GenericJsonlReader, JsonlZstWriter, change_suffix, find_files
from training_data_packer.utils.metadata import get_matching_part, get_metadata_value, read_metadata
from training_data_packer.utils.slurm import get_my_slurm_tasks


def package_file(
    src_file: Path, metadata: dict, contamination_file: Path, pii_file: Path, propella_file: Path, out_file: Path
) -> None:
    tmp_out_file = out_file.parent.joinpath("." + out_file.name)
    if out_file.exists():
        # File is already processed. Do not process it again
        logger.info(f"Skipping {out_file}, already exists")
        return
    if tmp_out_file.exists():
        logger.info(f"Remove old temporary file {tmp_out_file}")
        os.remove(tmp_out_file)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    contamination_filter = None
    block_filter = None
    pii_masker = None

    part_config, part_name = get_matching_part(metadata, src_file)

    src_reader = GenericJsonlReader(src_file)
    align_iter = AlignFieldNames(src_reader.read(), metadata)
    scrub_iter = field_scrubber_factory(align_iter, part_config)

    propella_data_iter = None
    if propella_file.exists():
        propella_data_iter = GenericJsonlReader(propella_file).read()
    propella_iter = propella_annotate_factory(scrub_iter, propella_data_iter)

    # After this comment are actual records removed. Processing cannot require zipping of dataset works.
    if "id" in metadata:
        pii_iter = AlignFieldNames(GenericJsonlReader(pii_file).read(), metadata, no_key_hierarchy=True)
        pii_masker = PIIMasker()
        pii_masked_iter = map(pii_masker.get_masker(pii_iter), propella_iter)

        contamination_ids = [x["id"] for x in AlignFieldNames(GenericJsonlReader(contamination_file).read(), metadata)]
        contamination_filter = FilterOnBlocklist("contamination", contamination_ids)
        filtered = contamination_filter.filter(pii_masked_iter)

        if "block" in part_config:
            block_filter = FilterOnBlocklist("block_list", part_config["block"])
            filtered = block_filter.filter(filtered)
    else:
        logger.info("No id field in metadata, skipping pii, decontamination and blocklist")
        filtered = scrub_iter
    if "wds+register" in metadata:
        logger.info("WDS+register files")
        filtered = itertools.chain.from_iterable(map(sample_register.process_record, filtered))
    sampled = sampler_factory(filtered, metadata, src_file, part_name)

    writer = JsonlZstWriter(tmp_out_file)
    writer.write(sampled)
    os.rename(tmp_out_file, out_file)
    metrics_collection = metrics.collect_metrics(
        src_reader,
        pii_masker,
        contamination_filter,
        block_filter,
        writer,
    )
    metrics_filename = out_file.parent.joinpath("." + out_file.name + ".metrics.json")
    metrics.write_metrics_to_file(metrics_collection, metrics_filename)


def process(collection_dir: Path, workers=1, slurm: bool = False, release: str | None = None) -> None:
    metadata = read_metadata(collection_dir.joinpath("metadata.yaml"))
    source_dir = collection_dir.joinpath(metadata["release"]["default"]["input"])
    contamination_dir = collection_dir.joinpath("nemo-curator")
    pii_dir = collection_dir.joinpath("openai-privacy-filter")
    propella_dir = collection_dir.joinpath("propella-4b")
    output_dir = collection_dir.joinpath("release-raw")

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
                contamination_file, pii_file, propella_file, out_file = _calculate_file_paths(
                    src_file, source_dir, contamination_dir, pii_dir, propella_dir, output_dir, metadata
                )

                job = executor.submit(
                    package_file,
                    src_file,
                    metadata,
                    contamination_file,
                    pii_file,
                    propella_file,
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
            contamination_file, pii_file, propella_file, out_file = _calculate_file_paths(
                src_file, source_dir, contamination_dir, pii_dir, propella_dir, output_dir, metadata
            )
            package_file(src_file, metadata, contamination_file, pii_file, propella_file, out_file)


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
    parser.add_argument("-r", "--release", help="Release to process, default is all")
    args = parser.parse_args()
    process(
        Path(args.collection_dir),
        workers=args.workers,
        slurm=args.slurm,
        release=args.release,
    )


if __name__ == "__main__":
    main()
