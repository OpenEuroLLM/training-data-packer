from typing import Any


def sample(doc: dict[str, Any], parameters) -> float:
    return float(doc["sample"])
