import importlib
import itertools
import random
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from loguru import logger

from training_data_packer.metadata.defaults import RUBBER_DEFAULT
from training_data_packer.metadata.metadata import Metadata, get_matching_part
from training_data_packer.processor.sample import sample_register


def read_sampler_fn(filename: Path | None):
    if filename is None:
        return lambda x: x
    spec = importlib.util.spec_from_file_location("dynamicsampler", filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dynamicsampler"] = module
    spec.loader.exec_module(module)
    return module.sample


def convert_to_type(dictionary: dict[str, str]) -> dict[str, str | int | float]:
    """Try to convert string dictionary values to int or float.

    Each value is converted to an integer if possible, otherwise to a float. If both
    conversions fail, the original value is kept.

    :param dictionary: dictionary with values to convert
    :returns: a new dictionary with the same keys and values converted to integers,
        floats, or left as strings
    """
    out = {}
    for k, value in dictionary.items():
        try:
            out[k] = int(value)
            continue
        except ValueError:
            pass
        try:
            out[k] = float(value)
            continue
        except ValueError:
            pass
        out[k] = value
    return out


class DynamicSampler:
    """Sample with a function from an external Python file.

    :param filename: path to the Python file with the sampler function
    :param sample_fn_parameters: parameters passed to the sampler function
    :param name: key used for the metrics
    """

    def __init__(self, filename, sample_fn_parameters, name="DynamicSampler"):
        self._sampler_ratio_fn = read_sampler_fn(filename)
        self._parameters = convert_to_type(sample_fn_parameters)
        self._name = name
        self._metric_lines_processed = 0
        self._metric_lines_produced = 0
        self._metric_lines_removed = 0
        self._metric_lines_upsampled = 0
        self._sampler_ratio_fn_exceptions = 0

    def get_metrics(self):
        return {
            self._name: {
                "document_read": self._metric_lines_processed,
                "document_written": self._metric_lines_produced,
                "document_removed": self._metric_lines_removed,
                "document_upsampled": self._metric_lines_upsampled,
                "sampler_ratio_exceptions": self._sampler_ratio_fn_exceptions,
            }
        }

    def get_mapper(self) -> Callable[dict[str, Any], list[dict[str, Any]]]:

        def mapper(record: dict[str, Any]) -> list[dict[str, Any]]:
            self._metric_lines_processed += 1
            try:
                multiplier = self._sampler_ratio_fn(record, self._parameters)
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
            elif new_lines > 1:
                self._metric_lines_upsampled += 1

            return result

        return mapper


def sampler_factory(
    data_iterator: Iterable[dict[str, Any]], metadata: Metadata, src_file_name: Path, section_name: str = "release"
) -> tuple[Iterable[dict[str, Any]], Any]:
    part_config, part_name = get_matching_part(metadata, src_file_name, section_name)
    if part_config is None or "sample" not in part_config:
        return data_iterator, None
    match part_config["sample"]:
        case "full":
            return data_iterator, None
        case "random":
            rubber = float(part_config.get("rubber", RUBBER_DEFAULT))
            fraction = min(float(part_config["budget"].strip("%")) / 100 * rubber, 1.0)
            return itertools.filterfalse(lambda x: random.random() > fraction, data_iterator), None
        case "wds+register":
            return itertools.chain.from_iterable(map(sample_register.process_record, data_iterator)), None
        case "dynamic":
            parameters = part_config["parameters"]
            filename = Path(part_config["filter"])
            if not filename.is_absolute():
                collection_dir = Path(metadata.get("_internal.collection_dir", None))
                filename = collection_dir.joinpath(filename)

            sampler = DynamicSampler(filename, parameters)
            return itertools.chain.from_iterable(map(sampler.get_mapper(), data_iterator)), sampler
        case _:
            logger.error(f"Unknown sampling rule {part_config['sample']} in {part_name}")
            raise ValueError(f"Unknown sampling rule {part_config['sample']} in {part_name}")
