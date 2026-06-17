import itertools
import os
from pathlib import Path

from loguru import logger

from training_data_packer.processor.clean import AlignFieldNames, field_scrubber_factory
from training_data_packer.processor.filters import FilterOnBlocklist
from training_data_packer.processor.pii_masking import PIIMasker
from training_data_packer.processor.propella import propella_annotate_factory
from training_data_packer.processor.sample import sample_register
from training_data_packer.processor.sample.sampler import sampler_factory
from training_data_packer.utils import metrics
from training_data_packer.utils.file import GenericJsonlReader, JsonlZstWriter
from training_data_packer.utils.metadata import get_matching_part


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
    sampled = sampler_factory(filtered, metadata, src_file)

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
