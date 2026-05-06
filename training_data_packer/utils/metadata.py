from pathlib import Path

import yaml
from loguru import logger


def get_matching_part(metadata: dict, src_file_name: Path) -> tuple[dict, str]:
    release = metadata["release"]
    if "default" in release:
        default_part = release["default"]
    else:
        default_part = {}
    for part in release:
        if part in str(src_file_name):
            if release[part] is None:
                release[part] = {}
            part_settings = default_part | release[part]
            logger.debug(f"Using part {part} for file {src_file_name} with settings {part_settings}")
            return part_settings, part
    logger.error(f"No part for file {src_file_name}")
    raise ValueError(f"No part for file {src_file_name}")


def read_metadata(file_path: Path) -> dict:
    with open(file_path) as file:
        metadata = yaml.safe_load(file)
        return metadata
