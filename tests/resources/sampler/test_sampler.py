from typing import Any


def get_sampling_ratio(doc: dict[str, Any], token_counts, sampling) -> float:
    return float(doc["sample"])
