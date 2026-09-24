import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

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
from training_data_packer.processor.filters import FilterOnBlocklist
from training_data_packer.processor.parallel_merger import ParallelLanguageMerger, ParallelSyntheticId
from training_data_packer.processor.pii_masking import PIIMasker, openai_mask_document
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
    """Schedule release packaging for files to be delivered according to metadata in collection dir."""
    metadata = read_metadata(collection_dir.joinpath("metadata.yaml"))
    metadata["_internal"]["mode"] = "release"
    source_dir = get_source_dir(metadata)
    src_suffix = get_in_suffix(metadata, "release")
    all_files = find_files(source_dir, src_suffix, part)
    if len(all_files) == 0:
        logger.error("No files detected, probably error in metadata.yaml")
        raise ValueError("No files detected")
    logger.info(f"Found {len(all_files)} files")

    schedule_files(all_files, metadata, package_file, workers, slurm)


def parallel_package_pipeline(
    src_iter: Iterable[dict[str, Any]],
    metadata: Metadata,
    part_config: dict[str, Any],
    piis: Iterable[dict[str, Any]],
    contaminations: Iterable[dict[str, Any]],
) -> tuple[Iterable[dict[str, Any]], list[dict[str, Any]]]:
    """Execute parallel package processing pipeline.

    Executes a parallel package processing pipeline that generates synthetic
    identifiers if required, applies filters based on PII and contamination
    blocklists, merges parallel language data components into documents, and
    collects performance metrics
    """
    if "id" not in metadata:
        parallel_synthetic_id = ParallelSyntheticId(metadata)
        synthetic_id_iter = parallel_synthetic_id.get_iterator(src_iter)
        pii_ids = {x["hash"] if "hash" in x else x["id"] for x in piis}
        contamination_ids = {x["hash"] if "hash" in x else x["id"] for x in contaminations}
    else:
        synthetic_id_iter = src_iter
        pii_ids = {x["id"] for x in piis}
        contamination_ids = {x["id"] for x in AlignFieldNames(contaminations, metadata)}

    pii_filter = FilterOnBlocklist("parallel-pii", pii_ids)
    contamination_filter = FilterOnBlocklist("parallel-contamination", contamination_ids)
    filtered_iter = contamination_filter.filter(pii_filter.filter(synthetic_id_iter))

    parallel_merger = ParallelLanguageMerger(metadata, part_config)
    return parallel_merger.get_merge_iterator(filtered_iter), [pii_filter, contamination_filter, parallel_merger]


def package_file(src_file: Path, metadata: Metadata) -> None:
    collection_dir = metadata["_internal.collection_dir"]
    part_config, _ = get_matching_part(metadata, src_file, section_name="release")
    if part_config is None:
        logger.info(f"Skipping {src_file}, does not match any release part")
        return

    out_file = calculate_file_path(src_file, metadata, collection_dir.joinpath("release-raw"))
    tmp_out_file, cont = prepare_output_file(out_file)
    if not cont:
        # File is already processed. Do not process it again
        logger.info(f"Skipping {out_file}, already exists")
        return

    # Decide what annotations to process
    annotations = part_config.get("annotations", [])
    if "nemo-curator" in annotations:
        contamination_file = calculate_file_path(src_file, metadata, collection_dir.joinpath("nemo-curator"))
    else:
        contamination_file = None
    if "openai-privacy-filter" in annotations:
        pii_file = calculate_file_path(src_file, metadata, collection_dir.joinpath("openai-privacy-filter"))
    else:
        pii_file = None

    contamination_filter = None
    block_filter = None
    pii_masker = None
    parallel_metrics = []

    is_parallel_text = metadata.get("_internal.parallel", False)

    src_reader = GenericJsonlReader(src_file)
    src_iter = src_reader.read()

    if is_parallel_text:
        align_iter, parallel_metrics = parallel_package_pipeline(
            src_iter,
            metadata,
            part_config,
            GenericJsonlReader(pii_file).read(),
            GenericJsonlReader(contamination_file).read(),
        )
    else:
        align_iter = AlignFieldNames(src_iter, metadata)

    scrub_iter = field_scrubber_factory(align_iter, part_config)

    # After this comment are actual records removed. Processing cannot require zipping of dataset works.
    if "id" in metadata:
        if not is_parallel_text:
            pii_iter = GenericJsonlReader(pii_file).read()
            pii_masker = PIIMasker(masker_fn=openai_mask_document, part_config=part_config)
            pii_masked_iter = map(pii_masker.get_masker(pii_iter), scrub_iter)

            contamination_ids = {
                x["id"] for x in AlignFieldNames(GenericJsonlReader(contamination_file).read(), metadata)
            }
            contamination_filter = FilterOnBlocklist("contamination", contamination_ids)
            filtered_iter = contamination_filter.filter(pii_masked_iter)
        else:
            filtered_iter = scrub_iter

        if "block" in part_config:
            block_filter = FilterOnBlocklist("block_list", part_config["block"])
            filtered_iter = block_filter.filter(filtered_iter)
    else:
        logger.info("No id field in metadata, skipping pii, decontamination and blocklist")
        filtered_iter = scrub_iter
    sampled, sampled_metrics = sampler_factory(filtered_iter, metadata, src_file)

    writer = JsonlZstWriter(tmp_out_file)
    writer.write(sampled)
    os.rename(tmp_out_file, out_file)
    metrics_collection = metrics.collect_metrics(
        src_reader,
        pii_masker,
        contamination_filter,
        block_filter,
        sampled_metrics,
        *parallel_metrics,
        writer,
    )
    metrics_filename = out_file.parent.joinpath("." + out_file.name + ".metrics.json")
    metrics.write_metrics_to_file(metrics_collection, metrics_filename)
