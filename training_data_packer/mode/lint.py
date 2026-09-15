from pathlib import Path
from typing import Any

from loguru import logger

from training_data_packer.metadata import Metadata, read_metadata
from training_data_packer.metadata.defaults import (
    DEFAULT_SUFFIX,
    PREFIX_DEFAULT,
    RUBBER_DEFAULT,
    SRC_LANGUAGE_DEFAULT,
    SRC_TEXT_DEFAULT,
    TGT_LANGUAGE_DEFAULT,
    TGT_TEXT_DEFAULT,
)
from training_data_packer.metadata.schema import Validator
from training_data_packer.processor.sample.sampler import read_sampler_fn
from training_data_packer.utils.file import GenericJsonlReader, find_files, get_subdirectories
from training_data_packer.utils.misc import get_dict_value


def process(collection_dir: Path) -> bool:
    metadata_file = collection_dir / "metadata.yaml"
    if not metadata_file.exists():
        logger.error(f"Metadata file does not exist in {collection_dir}")
        return False
    try:
        metadata = read_metadata(collection_dir.joinpath("metadata.yaml"))
        metadata["_internal"]["mode"] = "lint"
        _check_all_release_parts(metadata)
        if "source" in metadata:
            _check_all_source_parts(metadata)
        elif "openeurollm" in metadata:
            _check_all_source_parts(metadata, "openeurollm")
        else:
            logger.error("Section `source` or `openeurollm` missing.")
            ValueError("Section `source` or `openeurollm` missing.")
        if "uuid" in metadata:
            _check_uuid_section(metadata)
        if "sample" in metadata:
            _check_sample_section(metadata)
        if "propella-4b" in metadata:
            _check_propella_4b_section(metadata)
        if "nugget" in metadata:
            _check_all_source_parts(metadata, "nugget")
    except ValueError:
        return False
    return True


def _check_propella_4b_section(metadata: Metadata) -> None:
    """Validates the propella-4b section of the metadata."""
    section = "propella-4b"
    directory = Path(metadata.get("_internal.collection_dir")).joinpath(section)
    if not directory.exists():
        logger.info("Propella-4b data is not created.")


def _check_sample_section(metadata: Metadata) -> None:
    """Validates the sample section of the metadata."""
    section = "sample"
    fields = ["id", "text"]
    input = metadata[f"{section}.default.input"]
    if input not in metadata:
        raise ValueError(f"{section}.default.input references part `{input}` which is not defined.")
    _check_sections_files(metadata, section, fields)


def _check_uuid_section(metadata: Metadata) -> None:
    """Validates the uuid section of the metadata."""
    section = "uuid"
    fields = [metadata.get("id", "id")]
    input = metadata[f"{section}.default.input"]
    if input not in metadata:
        raise ValueError(f"{section}.default.input references part `{input}` which is not defined.")
    _check_sections_files(metadata, section, fields)


def _check_all_source_parts(metadata: Metadata, section: str = "source") -> None:
    if "parallel" in metadata:
        fields = [
            metadata.get("parallel.source.text", SRC_TEXT_DEFAULT),
            metadata.get("parallel.source.language", SRC_LANGUAGE_DEFAULT),
            metadata.get("parallel.target.text", TGT_TEXT_DEFAULT),
            metadata.get("parallel.target.language", TGT_LANGUAGE_DEFAULT),
        ]
    else:
        fields = []
        if "uuid" not in metadata:
            id_field = metadata.get("id", "id")
            fields.append(id_field)
        text_field = metadata.get("text", "text")
        fields.append(text_field)
    _check_sections_files(metadata, section, fields)


def _check_sections_files(metadata: Metadata, section: str, fields: list[Any]):
    directory = Path(metadata.get("_internal.collection_dir")).joinpath(section)
    if not directory.exists():
        logger.info(f"{section} data not created yet.")
    else:
        part_names = metadata.get_all_part_names(section)
        if len(part_names) == 0:
            raise ValueError(f"Section {section} has no parts defined.")
        for part in part_names:
            part_path = _build_part_path(section, part)
            part_settings = metadata.get_part(part_path)
            suffix = part_settings.get("suffix", metadata.get("suffix", DEFAULT_SUFFIX))
            record = _get_one_record_from_section_dir(directory, part, suffix)
            _check_fields_in_record(part, record, [fields])
        _check_non_matching_part_dirs(directory, part_names)


def _build_part_path(section: str, part_name: str) -> str:
    """Builds an escaped JSONpath string of section and part name."""
    return f'{section}."{part_name}"' if "'" in part_name else f"{section}.'{part_name}'"


def _check_non_matching_part_dirs(directory: Path, part_names: list[str]) -> bool:
    """Find if there are subdirectories not matching part names"""
    part_dirs = {p.name for p in get_subdirectories(directory)}
    exessive_parts = part_dirs - set(part_names)
    if len(exessive_parts) > 0:
        logger.warning(f"Dirs found in {directory.name} not matching parts: {', '.join(exessive_parts)}")
        return True
    return False


def _get_one_record_from_section_dir(directory: Path, part: str, suffix: Any) -> Any:
    part_files = find_files(directory, suffix, part)
    if len(part_files) == 0:
        logger.error(f"No source files found for part `{part}`. Expected in `{directory}` with suffix `{suffix}`.")
        raise ValueError(f"No source files found for part `{part}`. Expected in `{directory}` with suffix `{suffix}`.")
    first_row = next(GenericJsonlReader(part_files[0]).read())
    return first_row


def _check_fields_in_record(part: str, record: dict[str, Any], fields: list[str]):
    """
    Validates that the provided dictionary record contains all the specified fields
    and that these fields hold non-null values. If any field is missing or its value
    is None, an error is logged and a ValueError is raised.

    Args:
        part: String identifier for the logical section or part associated with
            the data, used in logging and error messages to aid debugging.
        record: Dictionary object containing the data to be validated.
        fields: List of string keys that must exist in the record.

    Raises:
        ValueError: If any key from the field list is not found in the record or
            maps to a None value.
    """
    for field in fields:
        if get_dict_value(record, field, None) is None:
            logger.error(f"First file for part {part} miss field {field}.")
            raise ValueError(f"First file for part {part} miss field {field}.")


def _check_all_release_parts(metadata: Metadata) -> None:
    """
    Performs validation and configuration checks for all parts under the release section
    in metadata

    :param metadata: metadata
    """
    for part in metadata.get_all_part_names("release"):
        part_path = _build_part_path("release", part)
        try:
            _check_release_part(part_path, metadata)
        except ValueError as e:
            logger.error(f"Part {part_path} failed validation. Reason: {e}")
            raise e


def _check_release_part(part_path: str, metadata: Metadata) -> None:
    """
    Performs validation and configuration checks for a specific part under the release section
    identified by its path.

    :param part_path: Path to part in metadata. Typically, `section.part-name`.
    :param metadata: metadata
    """
    part_settings = metadata.get_part(part_path)
    Validator().validate_release_part(part_settings)
    _check_sample_config(part_path, part_settings)
    _check_pack_config(part_path, part_settings)


def _check_pack_config(part_path: str, part_conf: dict[str, Any]):
    """
    Validates the pack configuration dictionary for a specific part based on the
    pack mode, ensuring that required keys are present and the configuration is
    valid for the specified mode.

    Args:
        part_path: Path to part in metadata. Typically, `section.part-name`.
        part_conf: A dictionary containing the configuration data, which must include a
            "pack" key indicating the mode and corresponding configuration options
            depending on that mode.

    Raises:
        ValueError: If validation fails

    Returns:
        None
    """
    match part_conf["pack"]:
        case "flat":
            if "prefix" not in part_conf:
                logger.warning(
                    f"In {part_path} setting `prefix` is recommended when `pack` has value `flat` to get unique names."
                    f"`prefix` default to `{PREFIX_DEFAULT}`."
                )
        case "tree":
            pass
        case _:
            raise ValueError(f"In {part_path} pack has an unknown value: {part_conf['pack']}")


def _check_sample_config(part_path: str, part_conf: dict[str, Any]):
    """
    Validates the sample configuration dictionary for a specific part based on the
    sampling mode, ensuring that required keys are present and the configuration is
    valid for the specified mode.

    Args:
        part_path: Path to part in metadata. Typically, `section.part-name`.
        part_conf: A dictionary containing the configuration data, which must include a
            "sample" key indicating the mode and corresponding configuration options
            depending on that mode.

    Raises:
        ValueError: If validation fails

    Returns:
        None
    """
    match part_conf["sample"]:
        case "full":
            pass
        case "random":
            if "rubber" not in part_conf:
                logger.warning(
                    f"In {part_path} setting `rubber` is recommended when `sample` has value `random`."
                    f"`rubber` default to `{RUBBER_DEFAULT}`."
                )
            if "budget" not in part_conf:
                raise ValueError(f"In {part_path} `sample` is `random` but `budget` is missing for part {part_path}.")
        case "dynamic":
            if "filter" not in part_conf:
                raise ValueError(f"In {part_path} sample is dynamic but filter is missing for part {part_path}.")
            if "parameters" not in part_conf:
                raise ValueError(f"In {part_path} sample is dynamic but parameters is missing for part {part_path}.")
            try:
                read_sampler_fn(part_conf["filter"])
            except Exception as e:
                raise ValueError(f"Fail to read sampler function for part {part_path}: {e}") from e
        case "wds+register":
            pass
        case _:
            raise ValueError(f"In {part_path} sample has an unknown value: {part_conf['sample']}")
