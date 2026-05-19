import itertools
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from loguru import logger

from training_data_packer import sample_register
from training_data_packer.utils.metadata import get_matching_part


def sampler_factory(
    data_iterator: Iterable[dict[str, Any]], metadata: dict, src_file_name: Path
) -> Iterable[dict[str, Any]]:
    if "release" not in metadata:
        return data_iterator
    release, release_name = get_matching_part(metadata, src_file_name)
    match release["sample"]:
        case "full":
            return data_iterator
        case "random":
            fraction = (float(release["budget"].strip("%")) + float(release["rubber"].strip("%"))) / 100
            return itertools.filterfalse(lambda x: random.random() > fraction, data_iterator)
        case "wds+register":
            return itertools.chain.from_iterable(map(sample_register.process_record, data_iterator))
        case _:
            logger.error(f"Unknown sampling rule {release['sample']} in {release_name}")
            raise ValueError(f"Unknown sampling rule {release['sample']} in {release_name}")
