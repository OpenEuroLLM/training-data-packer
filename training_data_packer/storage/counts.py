import json
from pathlib import Path
from typing import Any

from training_data_packer.utils.metadata import get_metadata_value


def get_counts(metadata: dict[str, Any], part_name: str) -> dict[str, Any]:
    collection_dir = Path(get_metadata_value(metadata, "_internal.collection_dir", None))
    counts_file = collection_dir.joinpath(f"counts/{part_name}/source.json")
    return json.loads(counts_file)
