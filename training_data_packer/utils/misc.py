import hashlib
from collections.abc import Callable

from loguru import logger


def hash_factory(hash_algo: str) -> Callable[str, str]:
    match hash_algo.lower():
        case "sha256":
            return lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()
        case _:
            logger.error(f"Unknown hash algorithm {hash_algo}")
            raise ValueError(f"Unknown hash algorithm {hash_algo}")
