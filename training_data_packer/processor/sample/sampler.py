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


def read_sampler_fn(filename: Path | None):
    if filename is None:
        return lambda x: x
    spec = importlib.util.spec_from_file_location("dynamicsampler", filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dynamicsampler"] = module
    spec.loader.exec_module(module)
    return module.sample


def convert_to_type(dictionary: dict[str, str]) -> dict[str, str | int | float]:
    """
    Attempts to convert the values of the input dictionary into appropriate
    numeric types. The function iterates through the key-value pairs of the
    provided dictionary and attempts to convert each value to an integer.
    If the conversion to an integer fails, it attempts to convert the value
    to a float. If both conversions raise a ValueError, the original value
    is retained in the output dictionary.

    Args:
        dictionary (dict): A dictionary containing the values to
            be converted. Keys can be of any hashable type, and values are
            expected to be strings or numeric representations suitable for
            conversion.

    Returns:
        dict: A new dictionary containing the same keys as the input, but
            with values converted to integers, floats, or left as strings
            based on their parseability.
    """
    out = {}
    for k in dictionary:
        value = dictionary[k]
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
                "lines_read": self._metric_lines_processed,
                "lines_written": self._metric_lines_produced,
                "lines_not_kept": self._metric_lines_removed,
                "lines_upsampled": self._metric_lines_upsampled,
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
    data_iterator: Iterable[dict[str, Any]], metadata: dict, src_file_name: Path, section_name: str = "release"
) -> tuple[Iterable[dict[str, Any]], Any]:
    part_config, part_name = get_matching_part(metadata, src_file_name, section_name)
    match part_config["sample"]:
        case "full":
            return data_iterator, None
        case "random":
            fraction = max(float(part_config["budget"].strip("%")) / 100 * float(part_config["rubber"]), 1.0)
            return itertools.filterfalse(lambda x: random.random() > fraction, data_iterator), None
        case "wds+register":
            return itertools.chain.from_iterable(map(sample_register.process_record, data_iterator)), None
        case "dynamic":
            parameters = part_config["parameters"]
            filename = Path(part_config["filter"])
            if not filename.is_absolute():
                collection_dir = Path(get_metadata_value(metadata, "_internal.collection_dir", None))
                filename = collection_dir.joinpath(filename)

            sampler = DynamicSampler(filename, parameters)
            return itertools.chain.from_iterable(map(sampler.get_mapper(), data_iterator)), sampler
        case _:
            logger.error(f"Unknown sampling rule {part_config['sample']} in {part_name}")
