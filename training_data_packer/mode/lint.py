from pathlib import Path

from loguru import logger

from training_data_packer.metadata import read_metadata


def process(collection_dir: Path) -> bool:
    metadata_file = collection_dir / "metadata.yaml"
    if not metadata_file.exists():
        logger.error("The metadata file does not exist")
        return False
    metadata = read_metadata(collection_dir.joinpath("metadata.yaml"))
    metadata["_internal"]["mode"] = "lint"

    return True
