import json
from pathlib import Path
from typing import Any


def collect_metrics(*objects) -> dict[str, Any]:
    metrics = {}
    for obj in objects:
        if obj is not None:
            metrics |= obj.get_metrics()
    return metrics


def write_metrics_to_file(metrics: dict[str, Any], metrics_file_name: Path) -> None:
    with open(metrics_file_name, mode="w") as file:
        json.dump(metrics, file, indent=4)


def aggregate_metrics(metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for m in metrics_list:
        for section_name, section_values in m.items():
            if section_name not in summary:
                summary[section_name] = {}
            for key, value in section_values.items():
                if isinstance(value, (int, float)):
                    summary[section_name][key] = summary[section_name].get(key, 0) + value
    return summary
