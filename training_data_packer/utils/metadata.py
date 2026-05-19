from pathlib import Path

import orjson as json
import yaml
from loguru import logger


def get_all_part_names(metadata: dict) -> list[str]:
    return sorted(filter(lambda x: x != "default", metadata["release"].keys()))


def get_shard_size_documents(part_config: dict) -> int:
    shard_size = part_config["shard"]
    if shard_size.endswith("bd"):
        return int(shard_size[:-2]) * 1_000_000_000
    if shard_size.endswith("md"):
        return int(shard_size[:-2]) * 1_000_000
    if shard_size.isdigit():
        return int(shard_size)
    raise ValueError(f"Invalid shard prefix {shard_size}")


def get_matching_part(metadata: dict, src_file_name: Path) -> tuple[dict, str]:
    release = metadata["release"]
    if "default" in release:
        default_part = release["default"]
    else:
        default_part = {}
    for part in release:
        if part in str(src_file_name):
            if release[part] is None or release[part] == "":
                release[part] = {}
            part_settings = default_part | release[part]
            logger.debug(f"Using part {part} for file {src_file_name} with settings {part_settings}")
            return part_settings, part
    logger.error(f"No part for file {src_file_name}")
    raise ValueError(f"No part for file {src_file_name}")


def read_metadata(file_path: Path) -> dict:
    with open(file_path) as file:
        metadata = yaml.load(file, Loader=yaml.BaseLoader)
        return metadata


def read_counts(file_path: Path) -> dict:
    with open(file_path) as file:
        return json.loads(file.read())
