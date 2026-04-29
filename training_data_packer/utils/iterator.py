import itertools
from collections.abc import Iterator
from typing import Any


def peek(iterator) -> tuple[Any, Iterator[Any]]:
    """
    Peek in an iterator.
    :param iterator: Iterator to peek into.
    :return: The first value in the iterator and a new iterator
        where the value is not consumed.
    """
    peek = next(iterator)
    return peek, itertools.chain([peek], iterator)

def get_until_key_change(iterator, key_fn) -> tuple[Any, list[Any], Iterator[Any]]:
    result = []
    item = next(iterator)
    result.append(item)
    key = key_fn(item)
    try:
        while item := next(iterator):
            if key == key_fn(item):
                result.append(item)
            else:
                return key, result, itertools.chain([item], iterator)
        raise RuntimeError()
    except StopIteration:
        return key, result, iter([])

