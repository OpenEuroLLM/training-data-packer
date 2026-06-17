import hashlib
from collections.abc import Callable

from iso639 import Lang
from loguru import logger


def hash_factory(hash_algo: str) -> Callable[str, str]:
    match hash_algo.lower():
        case "sha256":
            return lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()
        case "sha256-32":
            return lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()[:32]
        case _:
            logger.error(f"Unknown hash algorithm {hash_algo}")
            raise ValueError(f"Unknown hash algorithm {hash_algo}")


def lang_to_name(lang: str) -> str:
    """
    Retrieves the full language name corresponding to the provided language code or locale string.
    The input string is processed by splitting on underscores to isolate the primary language
    identifier, which is then used to retrieve the associated name attribute from the Lang
    enumeration or class.

    :param lang: The language code or locale string to process, typically containing the primary
                 language identifier and optionally a region tag separated by an underscore.
    :return: The descriptive name of the language corresponding to the primary code extracted from
             the input.
    """
    return Lang(lang.split("_")[0]).name
