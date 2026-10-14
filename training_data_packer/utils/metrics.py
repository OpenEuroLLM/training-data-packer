from pathlib import Path
from typing import Any

import orjson as json
from loguru import logger


def collect_metrics(*objects) -> dict[str, Any]:
    metrics = {}
    for obj in objects:
        if isinstance(obj, dict):
            metrics |= obj
        elif obj is not None:
            metrics |= obj.get_metrics()
    return metrics


def write_metrics_to_file(metrics: dict[str, Any], metrics_file_name: Path) -> None:
    with open(metrics_file_name, mode="wb") as f:
        f.write(json.dumps(metrics, option=json.OPT_APPEND_NEWLINE | json.OPT_INDENT_2, default=list))


def read_metrics_from_file(metrics_file_name: Path) -> dict[str, Any]:
    with open(metrics_file_name) as f:
        return json.loads(f.read())


def aggregate_metrics(metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for m in metrics_list:
        for section_name, section_values in m.items():
            if section_name not in summary:
                summary[section_name] = {}
            for key, value in section_values.items():
                if isinstance(value, (int, float)):
                    summary[section_name][key] = summary[section_name].get(key, 0) + value
                elif isinstance(value, str):
                    if key in summary[section_name]:
                        if summary[section_name][key] != value:
                            summary[section_name][key] = set(value).union(summary[section_name][key])
                        else:
                            pass  # If value is same as already collect, just keep as is.
                    else:
                        summary[section_name][key] = value
                elif isinstance(value, list):
                    if key in summary[section_name]:
                        summary[section_name][key] = set(value).union(summary[section_name][key])
                    else:
                        summary[section_name][key] = set(value)
                else:
                    logger.warning(f"Could not manage {section_name}.{key}: {value}")
    return summary


def merge_metrics(work_dir: Path, dest_dir: Path | None = None):
    def _collect_metrics_for_dir(output_file: Path):
        logger.info(f"Collecting metrics from {work_dir}")

        metrics_files = list(work_dir.rglob(".*.metrics.json"))
        logger.info(f"Found {len(metrics_files)} metrics files")

        all_metrics = []
        for metrics_file in metrics_files:
            try:
                with open(metrics_file) as f:
                    all_metrics.append(json.loads(f.read()))
            except Exception as e:
                logger.error(f"Failed to read {metrics_file}: {e}")

        if not all_metrics:
            logger.warning(f"No metrics files found in {work_dir}")
            return

        summary = aggregate_metrics(all_metrics)
        write_metrics_to_file(summary, output_file)
        logger.info(f"Summary written to {output_file}")

    if dest_dir is None:
        dest_dir = work_dir
    _collect_metrics_for_dir(dest_dir.joinpath("metrics.json"))
