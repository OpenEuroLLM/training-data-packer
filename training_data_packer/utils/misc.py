import hashlib
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import glom
import jsonpath_ng
from iso639 import Lang
from loguru import logger


def hash_factory(hash_algo: str) -> Callable[[str], str]:
    match hash_algo.lower():
        case "sha256":
            return lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()
        case "sha256-32":
            return lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()[:32]
        case _:
            logger.error(f"Unknown hash algorithm {hash_algo}")
            raise ValueError(f"Unknown hash algorithm {hash_algo}")


def lang_to_name(lang: str) -> str:
    """Get full language name from a language code.

    The primary language code is taken from the part before the first underscore.

    :param lang: language code or locale string, e.g. ``sv`` or ``sv_SE``
    :returns: the name of the language
    """
    return Lang(lang.split("_", maxsplit=1)[0]).name


def merge_hierarchy_dicts(dict_a, dict_b):
    """Merge two dictionaries recursively.

    Merge ``dict_a`` and ``dict_b`` by combining their keys and values, where ``dict_a``
    has precedence over ``dict_b``. Values that are not dictionaries are taken from
    ``dict_a``, unless it is None.

    :param dict_a: the dictionary with precedence
    :param dict_b: the dictionary to merge into ``dict_a``
    :returns: a new dictionary with the merged result
    """
    if isinstance(dict_a, dict) and isinstance(dict_b, dict):
        a_and_b = set(dict_a).intersection(dict_b)
        every_key = set(dict_a).union(dict_b)
        result = {
            k: merge_hierarchy_dicts(dict_a[k], dict_b[k])
            if k in a_and_b
            else deepcopy(dict_a[k] if k in dict_a else dict_b[k])
            for k in every_key
        }
        return result
    if dict_a is None:
        return deepcopy(dict_b)
    return deepcopy(dict_a)


def get_dict_value(dictionary: dict[str, Any], key: str, default: Any = None) -> Any:
    """Get one value matching a simplified JSONPath key or get default value.

    It supports array and dot notation in the key.

    :param dictionary: dictionary to search
    :param key: JSONPath expression identifying the value
    :param default: value to return if the expression has no match
    :returns: the value of the unique match, or the default if there is no match
    """
    key = key.replace("[", ".").replace("]", "")
    value = glom.glom(dictionary, key, default=default)
    return value


def get_dict_values(dictionary: dict[str, Any], key: str) -> Any:
    """Get all values matching the JSONPath key.

    :param dictionary: dictionary to search
    :param key: JSONPath expression identifying the values
    :returns: list of matching values, empty if there is no match
    """
    expr = jsonpath_ng.parse(key)
    match = expr.find(dictionary)
    if len(match) == 0:
        return []
    return [m.value for m in match]
