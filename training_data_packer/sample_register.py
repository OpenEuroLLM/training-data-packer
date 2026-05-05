import random
from typing import Any

REGISTERS = ["dtp", "HI", "HI-IN", "ID", "IN", "IP", "MT", "NA", "ne", "OP", "SP", "LY", "no-label"]

LABEL_HIERARCHY = {
    "MT": [], "LY": [], "SP": ["it"], "ID": [],
    "NA": ["ne", "sr", "nb"], "HI": ["re"],
    "IN": ["en", "ra", "dtp", "fi", "lt"],
    "OP": ["rv", "ob", "rs", "av"], "IP": ["ds", "ed"],
}
LABEL_PARENT = {c: p for p, cs in LABEL_HIERARCHY.items() for c in cs}

REGISTER_COEFF = {
    "dtp": 1.5,
    "HI": 1.5,
    "ID": 1.0,
    "IN": 1.0,
    "IP": 0.1,
    "MT": 0.1,
    "NA": 1.0,
    "ne": 1.0,
    "OP": 1.5,
    "SP": 1.0,
    "LY": 1.0,
    "no-label": 1.0,
    "HI-IN": 1.5
}

def _calc_wds_coeff(wds):
    """Polynomial for WDS scaling."""
    return (0.0213*(wds**3) - 0.4113*(wds**2) + 2.9626*wds - 7.1492)

def _assign_labels(probabilities, threshold):
    labels = set()
    for label, prob in probabilities.items():
        if prob >= threshold:
            labels.add(label)
            if label in LABEL_PARENT:
                labels.add(LABEL_PARENT[label])
    return labels

def _is_hybrid(labels):
    if len(labels) > 2:
        return True
    if len(labels) == 2:
        l1, l2 = list(labels)
        return not (
            (l1 in LABEL_PARENT and LABEL_PARENT[l1] == l2) or
            (l2 in LABEL_PARENT and LABEL_PARENT[l2] == l1)
        )
    return False

def process_record(record: dict[str, Any], length_limit: int = 200, threshold: float = 0.4, exclude_hybrids: bool = True) -> list[dict[str, Any]]:
    """
    Takes one record and decides whether it should be included or excluded. If included decide amount
    of upsampling.
    :param record: record to process
    :param length_limit: required length for text to be included
    :param threshold: Threshold on labels to be included.
    :param exclude_hybrids: Whether include with multiple labels or not.
    :return: list of records to be included
    """
    # 1. Length Filter
    if len(record.get("text", "")) < length_limit:
        record["delete"] = True
        return [record]

    # 2. Label Assignment
    probs = record.get("web-register", {})
    r = _assign_labels(probs, threshold)

    # 3. Hybrid Logic
    if len(r) == 0:
        register = "no-label"
    elif exclude_hybrids and _is_hybrid(r):
        if r == {"HI", "IN"}:
            register = "HI-IN"
        else:
            record["delete"] = True
            return [record]
    else:
        # Clean label logic
        selected = [j for j in r if j in REGISTERS]
        register = '-'.join(sorted(selected))

        # Normalize specific cases
        if register in ["NA-ne", "ne-NA"]: register = "ne"
        if register in ["IN-dtp", "dtp-IN"]: register = "dtp"

    if register not in REGISTER_COEFF:
        record["delete"] = True
        return [record]

    # 4. Sampling Logic
    doc_score = record.get("doc_scores", [0])[0]
    wds_coeff = _calc_wds_coeff(doc_score)
    multiplier = REGISTER_COEFF[register] * wds_coeff

    # 5. Writing (Upsampling/Downsampling)
    m_work = multiplier
    result = []
    while m_work >= 1:
        record["registers"] = list(r)
        record["upsampled"] = True
        result.append(record)
        m_work -= 1

    if random.random() < m_work:
        record["registers"] = list(r)
        record["upsampled"] = True
        result.append(record)

    return result
