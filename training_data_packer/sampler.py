import itertools
import random
from pathlib import Path

from loguru import logger

from training_data_packer.utils.metadata import get_matching_release


def sampler_factory(data_iterator, metadata: dict, src_file_name: Path):
    if "release" not in metadata:
        return data_iterator
    release, release_name = get_matching_release(metadata, src_file_name)
    match release["sample"]:
        case "full":
            return data_iterator
        case "random":
            fraction = float(release["budget"].strip("%"))/100
            return itertools.filterfalse(lambda x: random.random()>fraction, data_iterator)
        case "wds+register":
            # TODO: Implement WDS+register
            raise NotImplementedError
        case _:
            logger.error(f"Unknown sampling rule {release['sample']} in {release_name}")
            raise ValueError(f"Unknown sampling rule {release['sample']} in {release_name}")
