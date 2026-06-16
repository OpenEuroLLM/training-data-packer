import importlib
import itertools
import random
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from loguru import logger

from training_data_packer.processor.sample import sample_register
from training_data_packer.utils.metadata import get_matching_part, get_metadata_value


def read_sampler_fn(filename: Path | None) -> Callable[dict[str, Any], list[dict[str, Any]]]:
    if filename is None:
        return lambda x: x
    spec = importlib.util.spec_from_file_location("dynamicsampler", filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dynamicsampler"] = module
    spec.loader.exec_module(module)
    return module.sample


class DynamicSampler:
    def __init__(self, filename, name="DynamicSampler"):
        self._sampler_fn = read_sampler_fn(filename)
        self._name = name
        self._metric_lines_processed = 0
        self._metric_lines_produced = 0
        self._metric_lines_removed = 0

    def get_metrics(self):
        return {
            self._name: {
                "lines_read": self._metric_lines_processed,
                "lines_written": self._metric_lines_produced,
                "lines_not_keept": self._metric_lines_removed,
            }
        }

    def get_mapper(self) -> Callable[dict[str, Any], list[dict[str, Any]]]:

        def mapper(record: dict[str, Any]) -> list[dict[str, Any]]:
            self._metric_lines_processed += 1
            multiplier = self._sampler_fn(record)  # This method probably need some metadata and counts provided
            result = []
            while multiplier >= 1:
                result.append(record)
                multiplier -= 1

            if random.random() < multiplier:
                result.append(record)

            new_lines = len(result)
            self._metric_lines_produced += new_lines
            if new_lines == 0:
                self._metric_lines_removed += 1

            return result

        return mapper


def sampler_factory(
    data_iterator: Iterable[dict[str, Any]], metadata: dict, src_file_name: Path, part_name: str
) -> Iterable[dict[str, Any]]:
    if "release" not in metadata:
        return data_iterator
    release, release_name = get_matching_part(metadata, src_file_name)
    match release["sample"]:
        case "full":
            return data_iterator
        case "random":
            fraction = max(float(release["budget"].strip("%")) / 100 * float(release["rubber"]), 1.0)
            return itertools.filterfalse(lambda x: random.random() > fraction, data_iterator)
        case "wds+register":
            return itertools.chain.from_iterable(map(sample_register.process_record, data_iterator))
        case "sampler_fn":
            filename = Path(
                get_metadata_value(
                    metadata,
                    f"release.{part_name}.sample_fn_file",
                    get_metadata_value(metadata, "release.default.sample_fn_file", None),
                )
            )
            if not filename.is_absolute():
                filename = Path(get_metadata_value(metadata, "_internal.collection_dir", None)).joinpath(filename)
            sampler = DynamicSampler(filename)
            return itertools.chain.from_iterable(map(sampler.get_mapper(), data_iterator))
        case _:
            logger.error(f"Unknown sampling rule {release['sample']} in {release_name}")
            raise ValueError(f"Unknown sampling rule {release['sample']} in {release_name}")
