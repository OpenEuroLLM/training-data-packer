import os
from argparse import ArgumentError
from pathlib import Path

from loguru import logger

from training_data_packer.processor.clean import AlignFieldNames, field_scrubber_factory
from training_data_packer.processor.propella import propella_annotate_factory
from training_data_packer.processor.sample.sampler import sampler_factory
from training_data_packer.utils import metrics
from training_data_packer.utils.file import GenericJsonlReader, JsonlZstWriter
from training_data_packer.utils.metadata import get_matching_part


def sample_file(src_file: Path, metadata: dict, propella_file: Path, out_file: Path) -> None:
    tmp_out_file = out_file.parent.joinpath("." + out_file.name)
    if out_file.exists():
        # File is already processed. Do not process it again
        logger.info(f"Skipping {out_file}, already exists")
        return
    if tmp_out_file.exists():
        logger.info(f"Remove old temporary file {tmp_out_file}")
        os.remove(tmp_out_file)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    part_config, part_name = get_matching_part(metadata, src_file, section_name="sample")
    if part_config is None:
        logger.info(f"Skipping {src_file}, does not match a release part")
        return

    src_reader = GenericJsonlReader(src_file)
    align_iter = AlignFieldNames(src_reader.read(), metadata)
    scrub_iter = field_scrubber_factory(align_iter, part_config)

    propella_data_iter = None
    if propella_file.exists():
        propella_data_iter = GenericJsonlReader(propella_file).read()
    propella_iter = propella_annotate_factory(scrub_iter, propella_data_iter)

    if "sample" in metadata:
        sampled_iter, sampler_metrics = sampler_factory(propella_iter, metadata, src_file, section_name="sample")
    else:
        raise ArgumentError("Metadata does not contain sample section.")

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
