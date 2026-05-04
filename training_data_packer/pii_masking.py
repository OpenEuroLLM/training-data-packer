import ipaddress
import random
import string
from typing import Any

from loguru import logger

from training_data_packer.utils.iterator import get_until_key_change

# RFC 1918 Private Ranges
IPV4_PRIVATE_BLOCKS = [
    ipaddress.IPv4Network('10.0.0.0/8'),
    ipaddress.IPv4Network('172.16.0.0/12'),
    ipaddress.IPv4Network('192.168.0.0/16')
]

def _has_overlapping_ranges(pii_records: list[dict[str, Any]]) -> bool:
    """
    Check if the pii_records has overlapping ranges-
    :param pii_records: PII records, sorted from the last in doc to the first.
    :return: True if any pii_records has overlapping ranges, False otherwise.
    """
    for k, pii_record in enumerate(pii_records[1:]):
        if pii_record["end_pos"] > pii_records[k]["start_pos"]:
            return True
    return False


def _merge_overlapping_ranges(pii_records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], set[str]]:
    """
    Merge overlapping ranges of pii_records
    :param pii_records: Reverse on position on sorted list.
    :return: Pii records where overlaps are merged and set of types overlapping.
    """
    if len(pii_records) <= 1:
        return pii_records, set()
    result = []
    current = pii_records[0]
    merged_types = set()
    for pii_record in pii_records[1:]:
        if pii_record["end_pos"] > current["start_pos"]:
            current["value"] = pii_record["value"][:-(pii_record["end_pos"]-current["start_pos"])] + current["value"]
            current["start_pos"] = pii_record["start_pos"]
            if current["name"] != "MERGED":
                merged_types.add(current["name"])
            merged_types.add(pii_record["name"])
            current["name"] = "MERGED"
        else:
            result.append(current)
            current = pii_record
    result.append(current)
    return result, merged_types

def _remove_duplicates_inplace(records: list[Any]) -> list[Any]:
    write_index = 1

    for i in range(1, len(records)):
        if records[i] != records[i - 1]:
            records[write_index] = records[i]
            write_index += 1

    del records[write_index:]
    return records

def _replace_segment(text: str, start_pos: int, end_pos: int, new_segment: str) -> str:
    length = len(text)
    if start_pos >= length:
        raise ValueError(f"Start position outside text range {start_pos}-{end_pos}")
    if end_pos > length:
        raise ValueError(f"End position outside text range {start_pos}-{end_pos}")
    if start_pos > end_pos:
        raise ValueError(f"Start position after end position {start_pos}-{end_pos}")
    return text[:start_pos] + new_segment + text[end_pos:]

def _scramble_string(text: str) -> str:
    while True:
        # This loop ensure input text and scrambled string is not the same.
        result = []

        for char in text:
            if char.isalpha():
                if char.isupper():
                    result.append(random.choice(string.ascii_uppercase))
                else:
                    result.append(random.choice(string.ascii_lowercase))
            elif char.isdigit():
                result.append(random.choice(string.digits))
            else:
                result.append(char)
        result_str = "".join(result)
        if result_str != text:
            return result_str

def _scramble_ip_address(text: str) -> str:
    """
    Identifies if an IP is v4 or v6 and replaces it with a
    random address from the respective private/local range.
    """
    if text == "127.0.0.1" or text == "::1":
        return text
    try:
        addr = ipaddress.ip_address(text)

        if addr.version == 4:
            selected_block = random.choice(IPV4_PRIVATE_BLOCKS)
            random_index = random.randint(0, selected_block.num_addresses - 1)
            return str(selected_block[random_index])

        elif addr.version == 6:
            # Unique Local Address (ULA) range fd00::/8
            network_prefix = 0xfd00 << 112
            random_suffix = random.getrandbits(120)
            return str(ipaddress.IPv6Address(network_prefix | random_suffix))

        raise ValueError()
    except ValueError as e:
        raise ValueError("Invalid IP Address") from e

def _mask_email_address(document: dict[str, Any], pii_record: dict[str, Any]) -> dict[str, Any]:
    # TODO: random before @
    email_mask = "test@example.com"
    document["text"] = _replace_segment(document["text"], pii_record["start_pos"], pii_record["end_pos"], email_mask)
    return document

def _mask_with_scrambled_string(document: dict[str, Any], pii_record: dict[str, Any]) -> dict[str, Any]:
    scrambled_value = _scramble_string(pii_record["value"])
    document["text"] = _replace_segment(document["text"], pii_record["start_pos"], pii_record["end_pos"], scrambled_value)
    return document

def _mask_bitcoin_address(document: dict[str, Any], pii_record: dict[str, Any]) -> dict[str, Any]:
    if pii_record["value"][0] == "1" or pii_record["value"][0] == "3":
        scrambled_bitcoin = pii_record["value"][0] + _scramble_string(pii_record["value"][1:])
    elif pii_record["value"][0:3] == "bc1":
        scrambled_bitcoin = pii_record["value"][0:3] + _scramble_string(pii_record["value"][3:])
    else:
        logger.warning(f"Unknown bitcoin address format {pii_record['value']} in document {pii_record['id']}, using scrambled string")
        scrambled_bitcoin = _scramble_string(pii_record["value"])
    document["text"] = _replace_segment(document["text"], pii_record["start_pos"], pii_record["end_pos"], scrambled_bitcoin)
    return document

def _mask_ip_address(document: dict[str, Any], pii_record: dict[str, Any]) -> dict[str, Any]:
    try:
        scrambled_ip = _scramble_ip_address(pii_record["value"])
    except ValueError:
        scrambled_ip = _scramble_string(pii_record["value"])
    document["text"] = _replace_segment(document["text"], pii_record["start_pos"], pii_record["end_pos"], scrambled_ip)
    return document


def mask_document(document: dict[str, Any], pii_records: list[dict[str, Any]]) -> dict[str, Any]:
    if document is None or pii_records is None:
        logger.error("Either document or pii_records is None")
        raise ValueError("Unknown input to mask_document")
    pii_records_sorted = sorted(pii_records, key=lambda x: -x["start_pos"])
    pii_records_sorted = _remove_duplicates_inplace(pii_records_sorted)
    if _has_overlapping_ranges(pii_records_sorted):
        pii_records_sorted, merged_types = _merge_overlapping_ranges(pii_records_sorted)
        logger.info(f"Document {document["id"]} has overlapping PII. Merging overlapping ranges. Merged types {merged_types}")
    try:
        for pii_record in pii_records_sorted:
            match pii_record["name"]:
                case "BANK_ACCOUNT" | "CREDIT_CARD" | "DRIVER_LICENSE" | "GOV_ID" | "LICENSE_PLATE" | "PHONE_NUMBER" | "MERGED":
                    document = _mask_with_scrambled_string(document, pii_record)
                case "BITCOIN_ADDRESS":
                    document = _mask_bitcoin_address(document, pii_record)
                case "EMAIL_ADDRESS":
                    document = _mask_email_address(document, pii_record)
                case "IP_ADDRESS":
                    document = _mask_ip_address(document, pii_record)
                case _:
                    logger.warning(
                        f"Unknown pii record type {pii_record['name']} in document {document['id']}, masked as scrambled string")
                    document = _mask_with_scrambled_string(document, pii_record)
                    document["pii_unknown"] = True
    except ValueError as e:
        logger.warning(f"Document {document['id']} has pii issues {e}")
    document["pii_masks"] = len(pii_records_sorted)
    return document


def pii_key(record):
    return record['id']

class PiiMasker:
    """
    Iterator over a source data set. Masking all docs according to
    pii_data iterator. Documents in src_data and pii_data must be in
    the same order.
    Records that have been masked get the field masked=True.
    """

    def __init__(self, src_data, pii_data):
        self._src_data = src_data
        self._pii_data = pii_data
        self._next_pii_doc_id, self._next_pii_docs = None, None

    def __iter__(self):
        return self

    def __next__(self):
        if self._next_pii_doc_id is None:
            self._get_next_pii_doc()
        try:
            next_src_doc = next(self._src_data)
        except StopIteration as e:
            if self._next_pii_doc_id is not None:
                raise ValueError() from e
            raise e
        if self._next_pii_doc_id is not None and self._next_pii_doc_id == next_src_doc["id"]:
            next_src_doc = mask_document(next_src_doc, self._next_pii_docs)
            self._get_next_pii_doc()
        return next_src_doc

    def _get_next_pii_doc(self):
        try:
            self._next_pii_doc_id, self._next_pii_docs, self._pii_data = get_until_key_change(self._pii_data, pii_key)
        except StopIteration:
            self._next_pii_doc_id, self._next_pii_docs, self._pii_data = None, None, iter([])
