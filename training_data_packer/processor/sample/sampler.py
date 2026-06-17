import importlib
import itertools
import random
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from loguru import logger

from training_data_packer.processor.sample import sample_register
from training_data_packer.storage.counts import get_counts
from training_data_packer.utils.metadata import get_matching_part, get_metadata_value


def read_sampler_fn(filename: Path | None) -> Callable[dict[str, Any], list[dict[str, Any]]]:
    if filename is None:
        return lambda x: x
    spec = importlib.util.spec_from_file_location("dynamicsampler", filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dynamicsampler"] = module
    spec.loader.exec_module(module)
    return module.get_sampling_ratio


class DynamicSampler:
    def __init__(self, filename, token_counts, name="DynamicSampler"):
        self._sampler_ratio_fn = read_sampler_fn(filename)
        self._token_counts = token_counts
        self._name = name
        self._metric_lines_processed = 0
        self._metric_lines_produced = 0
        self._metric_lines_removed = 0
        self._sampler_ratio_fn_exceptions = 0

    def get_metrics(self):
        return {
            self._name: {
                "lines_read": self._metric_lines_processed,
                "lines_written": self._metric_lines_produced,
                "lines_not_keept": self._metric_lines_removed,
                "sampler_ratio_exceptions": self._sampler_ratio_fn_exceptions,
            }
        }

    def get_mapper(self) -> Callable[dict[str, Any], list[dict[str, Any]]]:

        def mapper(record: dict[str, Any]) -> list[dict[str, Any]]:
            self._metric_lines_processed += 1
            try:
                multiplier = self._sampler_ratio_fn(record, self._token_counts, "ease_in_out")
            except Exception as e:
                logger.warning(f"DynamicSampler caught exception in get_sampling_ratio: {e}")
                self._sampler_ratio_fn_exceptions += 1
                multiplier = 1

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
    data_iterator: Iterable[dict[str, Any]], metadata: dict, src_file_name: Path
) -> tuple[Iterable[dict[str, Any]], Any]:
    if "release" not in metadata:
        return data_iterator, None
    part_config, part_name = get_matching_part(metadata, src_file_name)
    match part_config["sample"]:
        case "full":
            return data_iterator, None
        case "random":
            fraction = max(float(part_config["budget"].strip("%")) / 100 * float(part_config["rubber"]), 1.0)
            return itertools.filterfalse(lambda x: random.random() > fraction, data_iterator), None
        case "wds+register":
            return itertools.chain.from_iterable(map(sample_register.process_record, data_iterator)), None
        case "sampler_ratio_fn":
            token_counts = get_counts(metadata, part_name)["tokens"]

            filename = Path(part_config["sample_fn_file"])
            if not filename.is_absolute():
                collection_dir = Path(get_metadata_value(metadata, "_internal.collection_dir", None))
                filename = collection_dir.joinpath(filename)

            sampler = DynamicSampler(filename, token_counts)
            return itertools.chain.from_iterable(map(sampler.get_mapper(), data_iterator)), sampler
        case _:
            logger.error(f"Unknown sampling rule {part_config['sample']} in {part_name}")
            raise ValueError(f"Unknown sampling rule {part_config['sample']} in {part_name}")
