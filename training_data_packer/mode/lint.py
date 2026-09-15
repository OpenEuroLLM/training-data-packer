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
        check_all_release_parts(metadata)
        if "source" in metadata:
            check_all_source_parts(metadata)
        elif "openeurollm" in metadata:
            check_all_source_parts(metadata, "openeurollm")
        else:
            logger.error("Section `source` or `openeurollm` missing.")
            ValueError("Section `source` or `openeurollm` missing.")
        if "uuid" in metadata:
            check_uuid_section(metadata)
        if "sample" in metadata:
            check_sample_section(metadata)
        if "propella-4b" in metadata:
            check_propella_4b_section(metadata)
        if "nugget" in metadata:
            check_all_source_parts(metadata, "nugget")
    except ValueError:
        return False
    return True


def check_propella_4b_section(metadata: Metadata) -> None:
    """Validates the propella-4b section of the metadata."""
    section = "propella-4b"
    directory = Path(metadata.get("_internal.collection_dir")).joinpath(section)
    if not directory.exists():
        logger.info("Propella-4b data is not created.")


def check_sample_section(metadata: Metadata) -> None:
    """Validates the sample section of the metadata."""
    section = "sample"
    directory = Path(metadata.get("_internal.collection_dir")).joinpath(section)
    input = metadata["sample.default.input"]
    if input not in metadata:
        raise ValueError(f"sample.default.input references part `{input}` which is not defined.")
    if not directory.exists():
        logger.info("Sampling is not created yet.")
        return
    part_names = metadata.get_all_part_names("uuid")
    for part in part_names:
        part_path = _build_part_path(section, part)
        part_settings = metadata.get_part(part_path)
        suffix = part_settings.get("suffix", metadata.get("suffix", DEFAULT_SUFFIX))
        source = part_settings.get("source")
        if source not in metadata:
            ValueError(f"Sample section references source `{source}` which is not defined.")
        record = _get_one_record_from_section_dir(directory, part, suffix)
        _check_fields_in_record(part, record, ["id", "text"])
    _check_non_matching_part_dirs(directory, part_names)


def check_uuid_section(metadata: Metadata) -> None:
    """Validates the uuid section of the metadata."""
    section = "uuid"
    directory = Path(metadata.get("_internal.collection_dir")).joinpath(section)
    input = metadata["uuid.default.input"]
    if input not in metadata:
        raise ValueError(f"uuid.default.input references part `{input}` which is not defined.")
    if not directory.exists():
        logger.info("Synthetic UUID id not created yet.")
        return
    part_names = metadata.get_all_part_names("uuid")
    for part in part_names:
        part_path = _build_part_path(section, part)
        part_settings = metadata.get_part(part_path)
        suffix = part_settings.get("suffix", metadata.get("suffix", DEFAULT_SUFFIX))
        source = part_settings.get("source")
        if source not in metadata:
            ValueError(f"UUID section references source `{source}` which is not defined.")
        record = _get_one_record_from_section_dir(directory, part, suffix)
        _check_fields_in_record(part, record, [metadata.get("id", "id")])
    _check_non_matching_part_dirs(directory, part_names)


def check_all_source_parts(metadata: Metadata, section: str = "source") -> None:
    directory = Path(metadata.get("_internal.collection_dir")).joinpath(section)
    require_id = "uuid" not in metadata
    parallel = "parallel" in metadata
    part_names = metadata.get_all_part_names("release")
    if len(part_names) == 0:
        raise ValueError(f"Section {section} has no parts defined.")
    for part in part_names:
        part_path = _build_part_path(section, part)
        part_settings = metadata.get_part(part_path)
        suffix = part_settings.get("suffix", metadata.get("suffix", DEFAULT_SUFFIX))
        record = _get_one_record_from_section_dir(directory, part, suffix)
        if parallel:
            fields = [
                metadata.get("parallel.source.text", SRC_TEXT_DEFAULT),
                metadata.get("parallel.source.language", SRC_LANGUAGE_DEFAULT),
                metadata.get("parallel.target.text", TGT_TEXT_DEFAULT),
                metadata.get("parallel.target.language", TGT_LANGUAGE_DEFAULT),
            ]
        else:
            fields = []
            if require_id:
                id_field = metadata.get("id", "id")
                fields.append(id_field)
            text_field = metadata.get("text", "text")
            fields.append(text_field)
        _check_fields_in_record(part, record, fields)
    _check_non_matching_part_dirs(directory, part_names)


def _build_part_path(section: str, part: str) -> str:
    return f'{section}."{part}"' if "'" in part else f"{section}.'{part}'"


def _check_non_matching_part_dirs(directory: Path, part_names: list[str]):
    part_dirs = {p.name for p in get_subdirectories(directory)}
    exessive_parts = part_dirs - set(part_names)
    if len(exessive_parts) > 0:
        logger.warning(f"Dirs found in {directory.name} not matching parts: {', '.join(exessive_parts)}")


def _get_one_record_from_section_dir(directory: Path, part: str, suffix: Any) -> Any:
    part_files = find_files(directory, suffix, part)
    if len(part_files) == 0:
        logger.error(f"No source files found for part `{part}`. Expected in `{directory}` with suffix `{suffix}.")
        raise ValueError(f"No source files found for part `{part}`. Expected in `{directory}` with suffix `{suffix}.")
    first_row = next(GenericJsonlReader(part_files[0]).read())
    return first_row


def _check_fields_in_record(part: str, record: dict[str, Any], fields: list[str]):
    for field in fields:
        if get_dict_value(record, field, None) is None:
            logger.error(f"First file for part {part} has no field {field}.")
            raise ValueError(f"First file for part {part} has no field {field}.")


def check_all_release_parts(metadata: Metadata) -> None:
    for part in metadata.get_all_part_names("release"):
        if "'" in part:
            part_path = f'release."{part}"'
        else:
            part_path = f"release.'{part}'"
        try:
            _check_release_part(part_path, metadata)
        except ValueError as e:
            logger.error(f"Part {part_path} failed validation. Reason: {e}")
            raise e


def _check_release_part(part_path: str, metadata: Metadata) -> None:
    part_settings = metadata.get_part(part_path)
    Validator().validate_release_part(part_settings)
    _check_sample_config(part_path, part_settings)
    _check_pack_config(part_path, part_settings)


def _check_pack_config(name: str, part: dict[str, Any]):
    match part["pack"]:
        case "flat":
            if "prefix" not in part:
                logger.warning(
                    f"In {name} setting `prefix` is recommended when `pack` has value `flat` to get unique names."
                    f"`prefix` default to `{PREFIX_DEFAULT}`."
                )
        case "tree":
            pass
        case _:
            raise ValueError(f"In {name} pack has an unknown value: {part['pack']}")


def _check_sample_config(name: str, part: dict[str, Any]):
    match part["sample"]:
        case "full":
            pass
        case "random":
            if "rubber" not in part:
                logger.warning(
                    f"In {name} setting `rubber` is recommended when `sample` has value `random`."
                    f"`rubber` default to `{RUBBER_DEFAULT}`."
                )
            if "budget" not in part:
                raise ValueError(f"In {name} `sample` is `random` but `budget` is missing for part {name}.")
        case "dynamic":
            if "filter" not in part:
                raise ValueError(f"In {name} sample is dynamic but filter is missing for part {name}.")
            if "parameters" not in part:
                raise ValueError(f"In {name} sample is dynamic but parameters is missing for part {name}.")
            try:
                read_sampler_fn(part["filter"])
            except Exception as e:
                raise ValueError(f"Fail to read sampler function for part {name}: {e}") from e
        case "wds+register":
            pass
        case _:
            raise ValueError(f"In {name} sample has an unknown value: {part['sample']}")
