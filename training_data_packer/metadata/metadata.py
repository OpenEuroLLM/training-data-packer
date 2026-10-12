from collections import UserDict
from pathlib import Path
from typing import Any

import glom
import jsonpath_ng
from loguru import logger

from training_data_packer.metadata.defaults import DEFAULT_SUFFIX
from training_data_packer.utils.file import change_suffix
from training_data_packer.utils.misc import merge_hierarchy_dicts


class Metadata(UserDict):
    def __getitem__(self, key) -> Any:
        expr = jsonpath_ng.parse(key)
        match = expr.find(self.data)
        if len(match) == 0:
            raise KeyError(f"No such key {key}")
        elif len(match) > 1:
            raise ValueError(f"{key} return more than one match in metadata.")
        return match[0].value

    def __setitem__(self, key, value) -> None:
        key = key.replace("[", ".").replace("]", "")
        return glom.assign(self.data, key, value)

    def __delitem__(self, key):
        key = key.replace("[", ".").replace("]", "")
        return glom.delete(self.data, key)

    def get(self, key: str, default=None):
        """Get a value from the metadata.

        Retrieve a value using a dot-notation key for deep access. Array indices are supported
        both as ``[i]`` and ``.i``.

        :param key: path to the target value, in dot notation for nested access
        :param default: value to return if the key path does not exist in the metadata
        :returns: the value at the key path, or the default if the key is not resolved
        :raises ValueError: if the key matches more than one value
        """
        expr = jsonpath_ng.parse(key)
        match = expr.find(self.data)
        if len(match) == 0:
            return default
        elif len(match) > 1:
            raise ValueError(f"{key} return more than one match in metadata.")
        return match[0].value

    def get_all_part_names(self, section: str, include_path: bool = False) -> list[str]:
        """Return all part names from metadata.

        :param section: section to start looking for part names from
        :param include_path: include the section path in the returned names, e.g. "v3.train"
        :returns: list of part names
        """
        reserved_part_names = ["default"]

        def _get_section_parts(current_section):
            section_keys = self[current_section].keys()
            section_parts = set(filter(lambda x: x not in reserved_part_names, section_keys))
            input_src = self.get(f"{current_section}.default.input")
            if input_src is not None:
                return section_parts.union(_get_section_parts(input_src))
            return section_parts

        part_names = sorted(_get_section_parts(section))
        if include_path:
            part_names = [f'{section}."{name}"' if "'" in name else f"{section}.'{name}'" for name in part_names]
        return part_names

    def get_part(self, part_path: str) -> dict[str, Any]:
        """Get a part of the metadata, merged with the defaults of its section.

        :param part_path: dot-separated path to the part, typically ``section.part-name``
        :returns: the part configuration, merged with the default values for the section
        :raises ValueError: if the path matches more than one entry in the metadata
        """

        def _get_path_to_related_default(part_path: str) -> str:
            default_parser = jsonpath_ng.parse(part_path)
            default_parser.right = jsonpath_ng.jsonpath.Fields("default")
            return str(default_parser)

        part = self.get(part_path)
        if part is None:
            part = {}
        default = self.get(_get_path_to_related_default(part_path))
        if default is None:
            default = {}
        return merge_hierarchy_dicts(part, default)


def get_shard_size_documents(part_config: dict[str, Any]) -> int:
    """Return shard size in documents from part config.

    Interpret the suffixes ``bd``, ``md`` and ``d`` as billion, million and single documents.

    :param part_config: part config dictionary
    :returns: shard size in documents
    :raises ValueError: if the shard size has an unknown suffix
    """
    shard_size = part_config["shard"]
    if shard_size.endswith("bd"):
        return int(shard_size[:-2]) * 1_000_000_000
    if shard_size.endswith("md"):
        return int(shard_size[:-2]) * 1_000_000
    if shard_size.endswith("d"):
        return int(shard_size[:-1])
    raise ValueError(f"Invalid shard prefix {shard_size}")


def _get_pre_section_part(
    metadata: Metadata, src_file_name: Path, default_part_config, section_name: str
) -> tuple[None, None] | tuple[dict, str]:
    pre_section = metadata[section_name]
    for part in pre_section:
        if part in ["default"]:
            continue
        if f"/{part}/" in str(src_file_name):
            part_settings = default_part_config
            logger.debug(f"Using part {part} for file {src_file_name} with default, part identified in {section_name}")
            return part_settings, part

    if "default" in pre_section and section_name != "source":
        next_section = pre_section["default"]["input"]
        return _get_pre_section_part(metadata, src_file_name, default_part_config, next_section)
    logger.warning(f"No part for file {src_file_name}")
    return None, None


def get_matching_part(
    metadata: Metadata, src_file_name: Path, section_name: str = "release"
) -> tuple[None, None] | tuple[dict, str]:
    """Get matching part config and part name from metadata for given source file name.

    :param metadata: metadata to search
    :param src_file_name: source file name
    :param section_name: name of the section to look for parts in
    :returns: tuple of part config and part name, or ``(None, None)`` if no part matches
    """
    section = metadata[section_name]
    if "default" in section:
        default_part_config = section["default"]
    else:
        default_part_config = {}
    for part in section:
        if part in str(src_file_name):
            if section[part] is None or section[part] == "":
                section[part] = {}
            part_settings = merge_hierarchy_dicts(section[part], default_part_config)
            logger.debug(f"Using part {part} for file {src_file_name} with settings {part_settings}")
            return part_settings, part

    if section_name != "source":
        next_section = default_part_config["input"]
        return _get_pre_section_part(metadata, src_file_name, default_part_config, next_section)
    logger.warning(f"No part for file {src_file_name}")
    return None, None


def get_source_dir(metadata: Metadata) -> Path:
    mode = metadata["_internal"]["mode"]
    input_src = metadata[mode]["default"]["input"]
    if mode == "release" and "parallel" in metadata[mode]["default"]:
        metadata["_internal"]["parallel"] = True
    else:
        metadata["_internal"]["parallel"] = False
    return metadata["_internal"]["collection_dir"].joinpath(input_src)


def get_in_suffix(metadata: Metadata, mode: str) -> str:
    input_dir = metadata.get(f"{mode}.default.input", None)
    return metadata.get(f"{input_dir}.default.suffix", metadata.get("suffix", DEFAULT_SUFFIX))


def calculate_file_path(src_file: Path, metadata: Metadata, process_dir: Path) -> Path:
    mode = metadata["_internal"]["mode"]
    input_suffix = get_in_suffix(metadata, mode)
    rel_file_path = src_file.relative_to(get_source_dir(metadata))
    out_suffix = DEFAULT_SUFFIX
    return change_suffix(process_dir.joinpath(rel_file_path), input_suffix, out_suffix)
