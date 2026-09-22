import itertools
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
from training_data_packer.utils.file import GenericJsonlReader, find_files
from training_data_packer.utils.misc import get_dict_value, get_dict_values


def process(collection_dir: Path) -> bool:
    metadata_file = collection_dir / "metadata.yaml"
    try:
        if not metadata_file.exists():
            raise ValueError(f"Metadata file does not exist in {collection_dir}")
        metadata = read_metadata(collection_dir.joinpath("metadata.yaml"))
        metadata["_internal"]["mode"] = "lint"
        _check_all_release_parts(metadata)

        if "source" in metadata:
            _check_all_source_parts(metadata)
        elif "openeurollm" in metadata:
            _check_all_source_parts(metadata, "openeurollm")
        else:
            raise ValueError("Section `source` or `openeurollm` missing.")

        if "uuid" in metadata:
            _check_uuid_section(metadata)
        elif collection_dir.joinpath("uuid").is_dir():
            raise ValueError("UUID directory exist but no section in metadata.")

        if "sample" in metadata:
            _check_sample_section(metadata)
        elif collection_dir.joinpath("sample").is_dir():
            raise ValueError("Sample directory exist but no section in metadata.")

        for annotation in ["propella-4b", "nemo-curator", "openai-privacy-filter"]:
            _check_annotation_section(metadata, annotation)

        if "nugget" in metadata:
            _check_all_source_parts(metadata, "nugget")
        elif collection_dir.joinpath("nugget").is_dir():
            raise ValueError("nugget directory exist but no section in metadata.")
    except ValueError as e:
        logger.error(f"Metadata is invalid: {e}")
        return False
    return True


def _check_annotation_section(metadata: Metadata, annotation: str) -> None:
    """
    Validates annotation section of the metadata. An annotation section is a section
    referenced by other parts via the `annotations` field.

    Args:
        metadata: Metadata object to validate.
        annotation: Identifier for the specific annotation section to validate e.g., propella-4b.

    Raises:
        ValueError: If the annotation section is defined in the metadata but not used in
        any annotations field, or if a directory for the annotation exists on the filesystem
        without a corresponding definition in the metadata.
    """
    directory = Path(metadata.get("_internal.collection_dir")).joinpath(annotation)
    if annotation in metadata:
        if not directory.exists():
            logger.warning(f"`{annotation}` data not created.")
        if annotation not in _get_all_annotations(metadata):
            raise ValueError(f"`{annotation}` section defined but not used in any annotations field.")
    elif directory.is_dir():
        raise ValueError(f"`{annotation}` directory exist but no section in metadata.")


def _check_sample_section(metadata: Metadata) -> None:
    """Validates the sample section of the metadata."""
    section = "sample"
    fields = ["id", "text"]
    input = metadata[f"{section}.default.input"]
    if input not in metadata:
        raise ValueError(f"{section}.default.input references part `{input}` which is not defined.")
    _check_sections_files(metadata, section, fields)
    for part in metadata.get_all_part_names(section):
        part_path = _build_part_path(section, part)
        part_settings = metadata.get_part(part_path)
        try:
            _check_annotations_config(part_path, part_settings, metadata, {"propella-4b"})
        except ValueError as e:
            logger.error(f"Part {part_path} failed validation. Reason: {e}")
            raise e


def _check_uuid_section(metadata: Metadata) -> None:
    """Validates the uuid section of the metadata."""
    section = "uuid"
    fields = [metadata.get("id", "id")]
    input = metadata[f"{section}.default.input"]
    if input not in metadata:
        raise ValueError(f"{section}.default.input references part `{input}` which is not defined.")
    _check_sections_files(metadata, section, fields)


def _check_all_source_parts(metadata: Metadata, section: str = "source") -> None:
    """Check all parts in a section that is source to other sections."""
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
        logger.warning(f"`{section}` data not created.")
    else:
        part_names = metadata.get_all_part_names(section)
        if len(part_names) == 0:
            raise ValueError(f"Section {section} has no parts defined.")
        for part in part_names:
            part_path = _build_part_path(section, part)
            part_settings = metadata.get_part(part_path)
            _check_annotations_config(
                part_path,
                part_settings,
                metadata,
                ["doc_scores", "web-register", "bsc-edu", "finepdfs-edu", "fineweb2-hq", "jql", "propella-4b"],
            )
            suffix = part_settings.get("suffix", metadata.get("suffix", DEFAULT_SUFFIX))
            record = _get_one_record_from_section_dir(directory, part, suffix)
            _check_fields_in_record(part, record, fields)
        _check_parts_and_dirs_match(directory, part_names)


def _build_part_path(section: str, part_name: str) -> str:
    """Builds an escaped JSONpath string of section and part name."""
    return f'{section}."{part_name}"' if "'" in part_name else f"{section}.'{part_name}'"


def _check_parts_and_dirs_match(directory: Path, part_names: list[str]) -> bool:
    """Find if there are subdirectories not matching part names"""
    part_dirs = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_dir()}

    allowed_dirs = set(part_names)
    for part_name in part_names:
        path_parts = Path(part_name)
        for i in range(len(path_parts.parts)):
            allowed_dirs.add(str(Path(*path_parts.parts[: i + 1])))

    excessive_dirs = part_dirs - allowed_dirs
    if len(excessive_dirs) > 0:
        logger.warning(
            f"Directories found in `{directory.name}` not matching parts: {', '.join(sorted(excessive_dirs))}"
        )
        return False

    excessive_parts = set(part_names) - part_dirs
    if len(excessive_parts) > 0:
        logger.warning(
            f"All parts does not correspond to a directory in `{directory.name}`: {', '.join(sorted(excessive_parts))}"
        )
        return False
    return True


def _get_one_record_from_section_dir(directory: Path, part: str, suffix: Any) -> Any:
    part_files = find_files(directory, suffix, part)
    if len(part_files) == 0:
        raise ValueError(f"No source files found for part `{part}`. Expected in `{directory}` with suffix `{suffix}`.")
    try:
        first_row = next(GenericJsonlReader(part_files[0]).read())
    except StopIteration as e:
        raise ValueError(f"Cannot read a record from {part_files[0]}") from e
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

    Raises:
        ValueError: If validation fails.

    Returns:
        None
    """
    part_settings = metadata.get_part(part_path)
    Validator().validate_release_part(part_settings)
    _check_annotations_config(part_path, part_settings, metadata, {"nemo-curator", "openai-privacy-filter"})
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
            if "prefix" not in part_conf or part_conf["prefix"].strip() == "":
                logger.warning(
                    f"In {part_path} setting `prefix` is recommended when `pack` has value `flat` to get unique names."
                    f" `prefix` default is `{PREFIX_DEFAULT}`."
                )
        case "tree":
            pass
        case _:
            raise ValueError(f"In {part_path} pack has an unknown value: {part_conf['pack']}")


def _check_no_sample_specific_fields(part_path: str, part_conf: dict[str, Any], invalid_fields: set[str]) -> None:
    """Ensures that only valid fields for the current sample mode are present."""
    found_invalid = invalid_fields & set(part_conf.keys())
    if found_invalid:
        raise ValueError(
            f"In {part_path}, with sample mode '{part_conf.get('sample')}', "
            f"the following fields should not be set: {', '.join(sorted(found_invalid))}"
        )


def _check_annotations_config(
    part_path: str, part_settings: dict[str, Any], metadata: Metadata, allowed_annotations: set[str]
) -> None:
    """
    Validates annotations defined in the part settings against a set of allowed
    values and ensures that each annotation references a valid section in the
    metadata.

    Args:
        part_path: Identifier for the part being checked, utilized to provide
            context in error messages.
        part_settings: Configuration containing the definitions to be validated,
            specifically looking for an 'annotations' key.
        metadata: Object acting as the source of truth for available sections,
            ensuring all annotations point to valid entries.
        allowed_annotations: Approved keys that are permitted for use in the
            configuration.

    Raises:
        ValueError: If the part configuration includes annotations that are not
            in the allowed set or if any annotation fails to map to a section in
            the metadata.

    Returns:
        None
    """
    part_annotations = set(part_settings.get("annotations", []))
    disallowed_annotations = part_annotations - allowed_annotations
    if len(disallowed_annotations) > 0:
        raise ValueError(f"In {part_path} annotations contain unknown value: {', '.join(disallowed_annotations)}")
    for pa in part_annotations:
        if pa not in metadata:
            raise ValueError(f"In {part_path} annotations {pa} miss corresponding section.")


def _check_sample_config(part_path: str, part_conf: dict[str, Any]) -> bool:
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
        True if sample configuration appears consistent.
    """
    match part_conf["sample"]:
        case "full":
            _check_no_sample_specific_fields(part_path, part_conf, {"budget", "rubber", "filter", "parameters"})
        case "random":
            _check_no_sample_specific_fields(part_path, part_conf, {"filter", "parameters"})
            if "rubber" not in part_conf:
                logger.warning(
                    f"In {part_path} setting `rubber` is recommended when `sample` has value `random`."
                    f" `rubber` default is `{RUBBER_DEFAULT}`."
                )
            if "budget" not in part_conf:
                raise ValueError(f"In {part_path} `sample` is `random` but `budget` is missing for part {part_path}.")
        case "dynamic":
            _check_no_sample_specific_fields(part_path, part_conf, {"rubber", "budget"})
            if "filter" not in part_conf:
                raise ValueError(f"In {part_path} sample is dynamic but filter is missing for part {part_path}.")
            if "parameters" not in part_conf:
                raise ValueError(f"In {part_path} sample is dynamic but parameters is missing for part {part_path}.")
            try:
                read_sampler_fn(part_conf["filter"])
            except Exception as e:
                raise ValueError(f"Fail to read sampler function for part {part_path}: {e}") from e
        case "wds+register":
            _check_no_sample_specific_fields(part_path, part_conf, {"budget", "rubber", "filter", "parameters"})
        case _:
            raise ValueError(f"In {part_path} sample has an unknown value: {part_conf['sample']}")
    return True


def _get_all_annotations(metadata: Metadata) -> set[str]:
    return set(itertools.chain.from_iterable(get_dict_values(metadata.data, "$..annotations")))
