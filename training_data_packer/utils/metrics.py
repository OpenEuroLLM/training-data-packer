import json
from pathlib import Path
from typing import Any, Dict


def collect_metrics(*objects) -> Dict[str, Any]:
    metrics = {}
    for obj in objects:
        if obj is not None:
            metrics |= obj.get_metrics()
    return metrics


def write_metrics_to_file(metrics: Dict[str, Any], metrics_file_name: Path) -> None:
    with open(metrics_file_name, mode="w") as file:
        json.dump(metrics, file, indent=4)
