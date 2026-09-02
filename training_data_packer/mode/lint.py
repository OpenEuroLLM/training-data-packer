from pathlib import Path

from loguru import logger

from training_data_packer.metadata import read_metadata
from training_data_packer.metadata.schema import Validator


def process(collection_dir: Path) -> bool:
    metadata_file = collection_dir / "metadata.yaml"
    if not metadata_file.exists():
        logger.error("The metadata file does not exist")
        return False
    metadata = read_metadata(collection_dir.joinpath("metadata.yaml"))
    metadata["_internal"]["mode"] = "lint"

    validator = Validator()
    result, error = validator.validate_metadata(metadata)
    if not result:
        logger.error(f"Schema validation failed: {error}")
        return False

    return True
