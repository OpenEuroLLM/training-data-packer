from pathlib import Path

from loguru import logger


def get_matching_release(metadata: dict, src_file_name: Path) -> tuple[dict, str]:
    releases = metadata["release"]
    for release in releases:
        if release in str(src_file_name):
            logger.info(f"Using release {release} for file {src_file_name}")
            return release[release], release
    if "default" in releases:
        logger.info(f"Using default release for file {src_file_name}")
        return releases["default"], "default"
    logger.error(f"No release for file {src_file_name}")
    raise ValueError(f"No release for file {src_file_name}")
