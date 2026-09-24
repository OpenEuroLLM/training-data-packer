import hashlib
from pathlib import Path

import yaml
from loguru import logger
from yaml import SafeLoader

from training_data_packer.metadata import Metadata
from training_data_packer.metadata.schema import Validator


def read_metadata(file_path: Path, log_content: bool = True) -> Metadata:
    """Read metadata from file.

    Reads metadata from file and returns it as Metadata object.
    All field values are strings.
    :param file_path: Path to metadata file.
    :param log_content: Log metadata read.
    :return: Metadata dictionary.
    """
    with open(file_path) as file:
        data = file.read()
        sha256_data = hashlib.sha256(data.encode("utf-8")).hexdigest()
        logger.info(f"Metadata sha256:{sha256_data}")
        if log_content:
            logger.info(f"Metadata content {file_path}:\n{data}\n")
        metadata_dict = yaml.load(data, Loader=SafeLoader)
        metadata_dict["_internal"] = {"collection_dir": file_path.parent, "sha256": sha256_data}
        metadata = Metadata(metadata_dict)
        try:
            Validator().validate_metadata(metadata)
        except ValueError as e:
            logger.error(f"Metadata validation failed: {e}")
            raise ValueError(e) from e
        return metadata
