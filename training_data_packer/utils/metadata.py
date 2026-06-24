from pathlib import Path
from typing import Any

import glom
import yaml
from loguru import logger


def get_metadata_value(metadata: dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Retrieve a specific value from the provided metadata structure using a
    dot-notation key for deep access.
    :param metadata: The source data structure, typically a dictionary, which
                     contains the data to be searched.
    :param key: A string representing the path to the target value, supporting
                dot notation for nested access.
    :param default: An optional fallback value to be returned if the specified
                    key path does not exist within the metadata.
    :return: The value extracted from the metadata corresponding to the given
             key, or the default value if the key is not resolved.
    """
    return glom.glom(metadata, key, default=default)


def get_all_part_names(metadata: dict[str, Any]) -> list[str]:
    """
    Returns all part names from metadata.
    :param metadata: Metadata dictionary.
    :return: List of part names.
    """
    return sorted(filter(lambda x: x != "default", metadata["source"].keys()))


def get_shard_size_documents(part_config: dict[str, Any]) -> int:
    """
    Returns shard size in documents from part config.
    Interprets extensions bd and md, billion and million documents.
    :param part_config: Part config dictionary.
    :return: Shard size in documents.
    """
    shard_size = part_config["shard"]
    if shard_size.endswith("bd"):
        return int(shard_size[:-2]) * 1_000_000_000
    if shard_size.endswith("md"):
        return int(shard_size[:-2]) * 1_000_000
    if shard_size.isdigit():
        return int(shard_size)
    raise ValueError(f"Invalid shard prefix {shard_size}")


def get_matching_part(metadata: dict[str, Any], src_file_name: Path, section_name: str = "release") -> tuple[dict, str]:
    """
    Returns matching part config and part name from metadata for given source file name.
    :param metadata: Metadata dictionary.
    :param src_file_name: Source file name.
    :param section_name: Name of section to looks for parts information. Defult is release.
    :return: Tuple of part config and part name.
    """
    section = metadata[section_name]
    if "default" in section:
        default_part = section["default"]
    else:
        default_part = {}
    for part in section:
        if part in str(src_file_name):
            if section[part] is None or section[part] == "":
                section[part] = {}
            part_settings = default_part | section[part]
            logger.debug(f"Using part {part} for file {src_file_name} with settings {part_settings}")
            return part_settings, part
    logger.error(f"No part for file {src_file_name}")
    raise ValueError(f"No part for file {src_file_name}")


def read_metadata(file_path: Path) -> dict[str, Any]:
    """
    Reads metadata from file and returns it as dictionary.
    All field values are strings.
    :param file_path: Path to metadata file.
    :return: Metadata dictionary.
    """
    with open(file_path) as file:
        # BaseLoader to guarantee that the YAML parser will return unicode strings
        metadata = yaml.load(file, Loader=yaml.BaseLoader)
        metadata["_internal"] = {"collection_dir": file_path.parent}
        return metadata
