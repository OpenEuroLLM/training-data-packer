import itertools
from pathlib import Path
from typing import Any

from loguru import logger

from training_data_packer.metadata import Metadata, read_metadata
from training_data_packer.metadata.defaults import (
    DEFAULT_ID,
    DEFAULT_SUFFIX,
    DEFAULT_TEXT,
    PREFIX_DEFAULT,
    RUBBER_DEFAULT,
    SRC_LANGUAGE_DEFAULT,
    SRC_TEXT_DEFAULT,
    TGT_LANGUAGE_DEFAULT,
    TGT_TEXT_DEFAULT,
)
from training_data_packer.metadata.schema import Validator
from training_data_packer.processor.count import get_tokenizer
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
            annotation_dir = Path(metadata.get("_internal.collection_dir")).joinpath(annotation)
            if annotation in metadata and annotation_dir.exists():
                _check_record_fields_in_sections_files(metadata, annotation, ["id"])

        for s in ["nemo-curator", "openai-privacy-filter"]:
            _check_input_in_section(metadata, s)

        if "nugget" in metadata:
            _check_all_source_parts(metadata, "nugget")
        elif collection_dir.joinpath("nugget").is_dir():
            raise ValueError("nugget directory exist but no section in metadata.")

        _check_tokenizer(metadata)
    except ValueError as e:
        logger.error(f"Metadata is invalid: {e}")
        return False
    return True


def _check_annotation_section(metadata: Metadata, annotation: str) -> None:
    """Validate annotation section.

    Validate an annotation section of the metadata. An annotation section is a section
    referenced by other parts via the ``annotations`` field.

    :param metadata: metadata to validate
    :param annotation: name of the annotation section to validate, e.g. ``propella-4b``
    :raises ValueError: if the annotation section is defined in the metadata but not used in
        any annotations field, or if a directory for the annotation exists on the filesystem
        without a corresponding definition in the metadata
    """
    directory = Path(metadata.get("_internal.collection_dir")).joinpath(annotation)
    if annotation in metadata:
        if not directory.exists():
            logger.warning(f"`{annotation}` data not created.")
        if annotation not in _get_all_annotations(metadata):
            raise ValueError(f"`{annotation}` section defined but not used in any annotations field.")
    elif directory.is_dir():
        raise ValueError(f"`{annotation}` directory exist but no section in metadata.")


def _check_input_in_section(metadata: Metadata, section: str) -> None:
    if section in Metadata:
        input = metadata[f"{section}.default.input"]
        if input not in metadata:
            raise ValueError(f"{section}.default.input references part `{input}` which is not defined.")


def _check_sample_section(metadata: Metadata) -> None:
    """Validate the sample section of the metadata.

    :param metadata: metadata to validate
    :raises ValueError: if the sample section or its parts are invalid
    """
    section = "sample"
    fields = [metadata.get("id", DEFAULT_ID), metadata.get("text", DEFAULT_TEXT)]
    _check_input_in_section(metadata, section)
    _check_record_fields_in_sections_files(metadata, section, fields)
    for part in metadata.get_all_part_names(section):
        part_path = _build_part_path(section, part)
        part_settings = metadata.get_part(part_path)
        try:
            _check_annotations_config(part_path, part_settings, metadata, {"propella-4b"})
        except ValueError as e:
            logger.error(f"Part {part_path} failed validation. Reason: {e}")
            raise e


def _check_uuid_section(metadata: Metadata) -> None:
    """Validate the uuid section of the metadata.

    :param metadata: metadata to validate
    :raises ValueError: if the uuid section is invalid
    """
    section = "uuid"
    fields = [metadata.get("id", DEFAULT_ID)]
    _check_input_in_section(metadata, section)
    _check_record_fields_in_sections_files(metadata, section, fields)


def _check_all_source_parts(metadata: Metadata, section: str = "source") -> None:
    """Check all parts in a section that is a source to another section.

    :param metadata: metadata to validate
    :param section: name of the section to check
    :raises ValueError: if a part has an unknown annotation or its records lack required fields
    """
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
            id_field = metadata.get("id", DEFAULT_ID)
            fields.append(id_field)
        text_field = metadata.get("text", DEFAULT_TEXT)
        fields.append(text_field)

    # Get all annotation fields used for any part
    optional_fields = set()
    part_names = metadata.get_all_part_names(section)
    for part in part_names:
        part_path = _build_part_path(section, part)
        part_settings = metadata.get_part(part_path)
        part_annotations = set(part_settings.get("annotations", []))

        disallowed_annotations = part_annotations - {
            "doc_scores",
            "web-register",
            "bsc-edu",
            "finepdfs-edu",
            "fineweb2-hq",
            "jql",
            "propella-4b",
        }
        if len(disallowed_annotations) > 0:
            raise ValueError(f"In {part_path} annotations contain unknown value: {', '.join(disallowed_annotations)}")

        optional_fields |= part_annotations

    _check_record_fields_in_sections_files(metadata, section, fields, optional_fields)


def _check_record_fields_in_sections_files(
    metadata: Metadata, section: str, required_fields: list[str], optional_fields: list[str] | None = None
) -> None:
    """Check if records match requirements in metadata.

    Work on a section, a subdirectory under collection_dir, and verify that:

    * for each part, the first record in the first non-empty file has all required fields
    * across all parts, each optional field is present in at least one of the checked records

    If an optional field does not exist, or the section directory does not exist, a warning is logged.

    :param metadata: metadata containing section definitions
    :param section: name of the section to validate
    :param required_fields: field names that must be present and not None in every checked record
    :param optional_fields: field names that should be present in at least one record across the section
    :raises ValueError: if a required field is missing or None in any checked record
    """
    directory = Path(metadata.get("_internal.collection_dir")).joinpath(section)
    if not directory.exists():
        logger.warning(f"`{section}` data not created.")
    else:
        part_names = metadata.get_all_part_names(section)
        if len(part_names) == 0:
            raise ValueError(f"Section {section} has no parts defined.")
        not_found_fields = set(optional_fields or {})
        for part in part_names:
            part_path = _build_part_path(section, part)
            part_settings = metadata.get_part(part_path)
            suffix = part_settings.get("suffix", metadata.get("suffix", DEFAULT_SUFFIX))
            record = _get_one_record_from_section_dir(directory, part, suffix)
            _check_fields_in_record(part, record, required_fields)

            record_keys = set(record.keys())
            not_found_fields -= record_keys

        if len(not_found_fields) > 0:
            logger.warning(
                f"{section} expected fields {', '.join(sorted(not_found_fields))} "
                f"in some records, not found in any sampled record"
            )
        _check_parts_and_dirs_match(directory, part_names)


def _build_part_path(section: str, part_name: str) -> str:
    """Build an escaped JSONPath string of section and part name.

    :param section: name of the section
    :param part_name: name of the part
    :returns: the escaped JSONPath to the part
    """
    return f'{section}."{part_name}"' if "'" in part_name else f"{section}.'{part_name}'"


def _check_parts_and_dirs_match(directory: Path, part_names: list[str]) -> bool:
    """Find if there are subdirectories not matching part names.

    Log a warning for each mismatch.

    :param directory: section directory to check
    :param part_names: names of the parts defined for the section
    :returns: True if directories and part names match, False otherwise
    """
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
    for part_file in part_files:
        try:
            first_row = next(GenericJsonlReader(part_file).read())
            return first_row
        except StopIteration:
            continue
    raise ValueError(
        f"Cannot read a record from any file for part `{part}`. "
        f"All files are empty: {', '.join(str(f) for f in part_files)}"
    )


def _check_fields_in_record(part: str, record: dict[str, Any], fields: list[str]):
    """Validate fields in a record.

    Validate that the record contains all the given fields and that they hold non-null
    values. If any field is missing or None, an error is logged.

    :param part: name of the part the record belongs to, used in log and error messages
    :param record: the record to validate
    :param fields: names of the fields that must exist in the record
    :raises ValueError: if any field is missing from the record or maps to None
    """
    for field in fields:
        if get_dict_value(record, field, None) is None:
            raise ValueError(f"First file for part {part} miss field {field}.")


def _check_all_release_parts(metadata: Metadata) -> None:
    """Validate all parts under release section.

    Perform validation and configuration checks for all parts under the release section
    in metadata.

    :param metadata: metadata to validate
    :raises ValueError: if any part fails validation
    """
    for part in metadata.get_all_part_names("release"):
        part_path = _build_part_path("release", part)
        try:
            _check_release_part(part_path, metadata)
        except ValueError as e:
            logger.error(f"Part {part_path} failed validation. Reason: {e}")
            raise e


def _check_release_part(part_path: str, metadata: Metadata) -> None:
    """Validate part under release section.

    Perform validation and configuration checks for a specific part under the release section
    identified by its path.

    :param part_path: path to the part in metadata, typically ``section.part-name``
    :param metadata: metadata to validate
    :raises ValueError: if validation fails
    """
    part_settings = metadata.get_part(part_path)
    Validator().validate_release_part(part_settings)
    _check_annotations_config(part_path, part_settings, metadata, {"nemo-curator", "openai-privacy-filter"})
    _check_sample_config(part_path, part_settings)
    _check_pack_config(part_path, part_settings)


def _check_pack_config(part_path: str, part_conf: dict[str, Any]):
    """Validate the pack configuration.

    Validate the pack configuration of a part based on the pack mode, ensuring that
    required keys are present and the configuration is valid for the mode.

    :param part_path: path to the part in metadata, typically ``section.part-name``
    :param part_conf: configuration of the part, with a "pack" key indicating the mode
        and the options for that mode
    :raises ValueError: if validation fails
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
    """Ensure only valid fields for the current sample mode are present.

    :param part_path: path to the part in metadata, typically ``section.part-name``
    :param part_conf: configuration of the part
    :param invalid_fields: fields that must not be set in the current sample mode
    :raises ValueError: if any of the invalid fields is set
    """
    found_invalid = invalid_fields & set(part_conf.keys())
    if found_invalid:
        raise ValueError(
            f"In {part_path}, with sample mode '{part_conf.get('sample')}', "
            f"the following fields should not be set: {', '.join(sorted(found_invalid))}"
        )


def _check_annotations_config(
    part_path: str, part_settings: dict[str, Any], metadata: Metadata, allowed_annotations: set[str]
) -> None:
    """Validate annotations.

    Validate the annotations of a part against a set of allowed values and ensure that
    each annotation references a valid section in the metadata.

    :param part_path: path to the part in metadata, used in error messages
    :param part_settings: configuration of the part, with an optional "annotations" key
    :param metadata: metadata holding the available sections
    :param allowed_annotations: annotations permitted for the part
    :raises ValueError: if the part has annotations that are not allowed, or if any
        annotation does not map to a section in the metadata
    """
    part_annotations = set(part_settings.get("annotations", []))
    disallowed_annotations = part_annotations - allowed_annotations
    if len(disallowed_annotations) > 0:
        raise ValueError(f"In {part_path} annotations contain unknown value: {', '.join(disallowed_annotations)}")
    for pa in part_annotations:
        if pa not in metadata:
            raise ValueError(f"In {part_path} annotations {pa} miss corresponding section.")


def _check_sample_config(part_path: str, part_conf: dict[str, Any]) -> bool:
    """Validate the sample configuration.

    Validate the sample configuration of a part based on the sampling mode, ensuring
    that required keys are present and the configuration is valid for the mode.

    :param part_path: path to the part in metadata, typically ``section.part-name``
    :param part_conf: configuration of the part, with a "sample" key indicating the mode
        and the options for that mode
    :returns: True if the sample configuration appears consistent
    :raises ValueError: if validation fails
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


def _check_tokenizer(metadata: Metadata):
    if "tokenizer" in metadata:
        try:
            get_tokenizer(metadata["tokenizer"])
        except Exception as e:
            ValueError(f"Cannot load tokenizer {metadata['tokenizer']}.", e)
