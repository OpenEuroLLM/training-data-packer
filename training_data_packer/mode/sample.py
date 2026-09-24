import os
from pathlib import Path

from loguru import logger

from training_data_packer.metadata import (
    Metadata,
    calculate_file_path,
    get_in_suffix,
    get_matching_part,
    get_source_dir,
    read_metadata,
)
from training_data_packer.processor.clean import AlignFieldNames, field_scrubber_factory
from training_data_packer.processor.propella import propella_annotate_factory
from training_data_packer.processor.sample.sampler import sampler_factory
from training_data_packer.utils import metrics
from training_data_packer.utils.file import GenericJsonlReader, JsonlZstWriter, find_files, prepare_output_file
from training_data_packer.utils.slurm import schedule_files


def process(
    collection_dir: Path,
    workers=1,
    slurm: bool = False,
    part: str | None = None,
) -> None:
    """Schedule sampling for files to be sampled according to metadata in collection dir."""
    metadata = read_metadata(collection_dir.joinpath("metadata.yaml"))
    metadata["_internal"]["mode"] = "sample"
    source_dir = get_source_dir(metadata)
    src_suffix = get_in_suffix(metadata, "sample")
    all_files = find_files(source_dir, src_suffix, part)
    if len(all_files) == 0:
        logger.error("No files detected, probably error in metadata.yaml")
        raise ValueError("No files detected")
    logger.info(f"Found {len(all_files)} files")

    schedule_files(all_files, metadata, sample_file, workers, slurm)


def sample_file(src_file: Path, metadata: Metadata) -> None:
    collection_dir = metadata["_internal.collection_dir"]
    part_config, _ = get_matching_part(metadata, src_file, section_name="sample")
    if part_config is None:
        logger.info(f"Skipping {src_file}, does not match a release part")
        return

    out_file = calculate_file_path(src_file, metadata, collection_dir.joinpath("sample"))
    tmp_out_file, cont = prepare_output_file(out_file)
    if not cont:
        # File is already processed. Do not process it again
        logger.info(f"Skipping {out_file}, already exists")
        return

    annotations = part_config.get("annotations", [])
    if "propella-4b" in annotations:
        propella_file = calculate_file_path(src_file, metadata, collection_dir.joinpath("propella-4b"))
    else:
        propella_file = None

    src_reader = GenericJsonlReader(src_file)
    align_iter = AlignFieldNames(src_reader.read(), metadata)
    scrub_iter = field_scrubber_factory(align_iter, part_config)

    propella_iter = propella_annotate_factory(scrub_iter, GenericJsonlReader(propella_file).read())

    if "sample" in metadata:
        sampled_iter, sampler_metrics = sampler_factory(propella_iter, metadata, src_file, section_name="sample")
    else:
        raise ValueError("Metadata does not contain sample section.")

    writer = JsonlZstWriter(tmp_out_file)
    writer.write(sampled_iter)
    os.rename(tmp_out_file, out_file)
    metrics_collection = metrics.collect_metrics(
        src_reader,
        sampler_metrics,
        writer,
    )
    metrics_filename = out_file.parent.joinpath("." + out_file.name + ".metrics.json")
    metrics.write_metrics_to_file(metrics_collection, metrics_filename)
