import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from loguru import logger

from training_data_packer.processor.clean import AlignFieldNames, field_scrubber_factory
from training_data_packer.processor.filters import FilterOnBlocklist
from training_data_packer.processor.parallel_merger import ParallelLanguageMerger, ParallelSyntheticId
from training_data_packer.processor.pii_masking import PIIMasker, openai_mask_document
from training_data_packer.processor.sample.sampler import sampler_factory
from training_data_packer.utils import metrics
from training_data_packer.utils.file import GenericJsonlReader, JsonlZstWriter
from training_data_packer.utils.metadata import get_matching_part, get_metadata_value


def parallel_package_pipeline(
    src_iter: Iterable[dict[str, Any]],
    metadata: dict[str, Any],
    part_config: dict[str, Any],
    piis: Iterable[dict[str, Any]],
    contaminations: Iterable[dict[str, Any]],
) -> Iterable[dict[str, Any]]:
    """
    Executes a parallel package processing pipeline that generates synthetic
    identifiers if required, applies filters based on PII and contamination
    blocklists, merges parallel language data components into documents, and
    collects performance metrics
    """
    if "id" not in metadata:
        parallel_synthetic_id = ParallelSyntheticId(metadata)
        synthetic_id_iter = parallel_synthetic_id.get_iterator(src_iter)
        pii_ids = {x["hash"] for x in piis}
        contamination_ids = {x["hash"] for x in contaminations}
    else:
        synthetic_id_iter = src_iter
        pii_ids = {x["id"] for x in piis}
        contamination_ids = {x["id"] for x in AlignFieldNames(contaminations, metadata)}

    pii_filter = FilterOnBlocklist("parallel-pii", pii_ids)
    contamination_filter = FilterOnBlocklist("parallel-contamination", contamination_ids)
    filtered_iter = contamination_filter.filter(pii_filter.filter(synthetic_id_iter))

    parallel_merger = ParallelLanguageMerger(metadata, part_config)
    return parallel_merger.get_merge_iterator(filtered_iter), [pii_filter, contamination_filter, parallel_merger]


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
    parallel_metrics = []

    part_config, part_name = get_matching_part(metadata, src_file, section_name="release")

    is_parallel_text = get_metadata_value(metadata, "_internal.parallel", False)

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
        *parallel_metrics,
        writer,
    )
    metrics_filename = out_file.parent.joinpath("." + out_file.name + ".metrics.json")
    metrics.write_metrics_to_file(metrics_collection, metrics_filename)
