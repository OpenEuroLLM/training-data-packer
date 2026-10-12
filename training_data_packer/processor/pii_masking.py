import ipaddress
import itertools
import random
import re
import string
from collections.abc import Callable, Iterable
from typing import Any

from loguru import logger

# RFC 1918 Private Ranges
IPV4_PRIVATE_BLOCKS = [
    ipaddress.IPv4Network("10.0.0.0/8"),
    ipaddress.IPv4Network("172.16.0.0/12"),
    ipaddress.IPv4Network("192.168.0.0/16"),
]


def _has_overlapping_ranges(pii_records: list[dict[str, Any]]) -> bool:
    """Check if the PII records have overlapping ranges.

    :param pii_records: PII records, sorted from the last in the document to the first
    :returns: True if any PII records have overlapping ranges, False otherwise
    """
    for k, pii_record in enumerate(pii_records[1:]):
        if pii_record["end_pos"] > pii_records[k]["start_pos"]:
            return True
    return False


def _merge_overlapping_ranges(
    pii_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge overlapping ranges of PII records.

    :param pii_records: PII records, sorted on position in reverse order
    :returns: PII records where overlapping records are merged
    """
    if len(pii_records) <= 1:
        return pii_records
    result = []
    current = pii_records[0]
    merged_types = set()
    for pii_record in pii_records[1:]:
        if pii_record["end_pos"] > current["start_pos"]:
            current["value"] = pii_record["value"][: -(pii_record["end_pos"] - current["start_pos"])] + current["value"]
            current["start_pos"] = pii_record["start_pos"]
            if current["name"] != "MERGED":
                merged_types.add(current["name"])
            merged_types.add(pii_record["name"])
            current["name"] = "MERGED"
        else:
            result.append(current)
            current = pii_record
    result.append(current)
    return result


def _remove_duplicates_inplace(records: list[Any]) -> list[Any]:
    """Remove duplicate items in sorted list.

    This exists to be able to remove duplicates when the list holds dictionaries, which are not hashable.

    :param records: sorted list of records
    :returns: the list without duplicates
    """
    write_index = 1

    for i in range(1, len(records)):
        if records[i] != records[i - 1]:
            records[write_index] = records[i]
            write_index += 1

    del records[write_index:]
    return records


def _replace_segment(text: str, start_pos: int, end_pos: int, new_segment: str) -> str:
    """Replace segment of text with new segment.

    :param text: original text
    :param start_pos: start position of the segment to replace
    :param end_pos: end position of the segment to replace
    :param new_segment: new segment to replace with
    :returns: text with the segment replaced
    :raises ValueError: if the positions are outside the text or start is after end
    """
    length = len(text)
    if start_pos >= length:
        raise ValueError(f"Start position outside text range {start_pos}-{end_pos}")
    if end_pos > length:
        raise ValueError(f"End position outside text range {start_pos}-{end_pos}")
    if start_pos > end_pos:
        raise ValueError(f"Start position after end position {start_pos}-{end_pos}")
    return text[:start_pos] + new_segment + text[end_pos:]


def _scramble_string(text: str) -> str:
    """Scramble string by replacing each character with a random character of the same type.

    :param text: string to scramble
    :returns: the scrambled string
    """
    count = 0
    while True:
        # This loop ensures input text and scrambled string are different.
        count += 1
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
        if result_str != text or count > 3:
            return result_str


def _scramble_ip_address(text: str) -> str:
    """Scramble IP address.

    Identify if an IP is v4 or v6 and replace it with a random address from the
    respective private/local range. Loopback addresses are returned unchanged.

    :param text: IP address to scramble
    :returns: the scrambled IP address
    :raises ValueError: if the text is not a valid IP address
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
            network_prefix = 0xFD00 << 112
            random_suffix = random.getrandbits(120)
            return str(ipaddress.IPv6Address(network_prefix | random_suffix))

        raise ValueError()
    except ValueError as e:
        raise ValueError("Invalid IP Address") from e


def _split_email(email):
    pattern = r"(_at_| at |\(a[t]?\)|{a[t]?}|@)"

    # Split the email string at the pattern, maximum of 1 split
    parts = re.split(pattern, email, maxsplit=1, flags=re.IGNORECASE)

    if len(parts) == 3:
        username = parts[0].strip()
        separator = parts[1]
        domain = parts[2].strip()
        return username, domain, separator
    else:
        raise ValueError(f"Invalid email {email}")


def _mask_email_address(document: dict[str, Any], pii_record: dict[str, Any], cache: dict[str, str]) -> dict[str, Any]:
    """Mask email address.

    Mask an email address by replacing it with a random email address.

    :param document: document to mask
    :param pii_record: PII record containing the position of the email address in the document
    :param cache: replacement cache, to replace the same value with the same scrambled text within a document
    :returns: the masked document
    """
    value = pii_record["value"]
    if value in cache:
        email_mask = cache[value]
    else:
        try:
            user_name, domain, separator = _split_email(value)
            if user_name != "" and domain != "":
                email_mask = f"{_scramble_string(user_name)}{separator}example.com"
            elif domain == "":
                email_mask = f"{_scramble_string(user_name)}{separator}"
            else:
                logger.warning(f"PII record {pii_record} has not a valid email. Scrambling as string.")
                email_mask = _scramble_string(value)
        except ValueError:
            logger.warning(f"PII record {pii_record} has not a valid email. Scrambling as string.")
            email_mask = _scramble_string(value)
        cache[value] = email_mask
    document["text"] = _replace_segment(document["text"], pii_record["start_pos"], pii_record["end_pos"], email_mask)
    return document


def _mask_with_scrambled_string(
    document: dict[str, Any], pii_record: dict[str, Any], cache: dict[str, str]
) -> dict[str, Any]:
    """Mask string.

    Mask a string in a document by replacing it with a scrambled string.

    :param document: document to mask
    :param pii_record: PII record containing the position of the string in the document
    :param cache: replacement cache, to replace the same value with the same scrambled text within a document
    :returns: the masked document
    """
    value = pii_record["value"]
    if value in cache:
        scrambled_value = cache[value]
    else:
        scrambled_value = _scramble_string(value)
        cache[value] = scrambled_value
    document["text"] = _replace_segment(
        document["text"], pii_record["start_pos"], pii_record["end_pos"], scrambled_value
    )
    return document


def _mask_bitcoin_address(document: dict[str, Any], pii_record: dict[str, Any], cache) -> dict[str, Any]:
    """Mask bitcoin address.

    Mask a bitcoin address in a document by replacing it with a scrambled string, keeping the bitcoin prefix.

    :param document: document to mask
    :param pii_record: PII record containing the position of the bitcoin address in the document
    :param cache: replacement cache, to replace the same value with the same scrambled text within a document
    :returns: the masked document
    """
    value = pii_record["value"]
    if value in cache:
        scrambled_bitcoin = cache[value]
    elif value[0] == "1" or value[0] == "3":
        scrambled_bitcoin = value[0] + _scramble_string(value[1:])
    elif value[0:3] == "bc1":
        scrambled_bitcoin = value[0:3] + _scramble_string(value[3:])
    else:
        logger.warning(f"Unknown bitcoin address format {value} in document {pii_record['id']}, using scrambled string")
        scrambled_bitcoin = _scramble_string(value)
        cache[value] = scrambled_bitcoin
    document["text"] = _replace_segment(
        document["text"],
        pii_record["start_pos"],
        pii_record["end_pos"],
        scrambled_bitcoin,
    )
    return document


def _mask_ip_address(document: dict[str, Any], pii_record: dict[str, Any], cache: dict[str, str]) -> dict[str, Any]:
    """Mask IP address in document.

    Mask an IP address in a document by replacing it with a scrambled IP address. Supports both IPv4 and IPv6.

    :param document: document to mask
    :param pii_record: PII record containing the position of the IP address in the document
    :param cache: replacement cache, to replace the same value with the same scrambled text within a document
    :returns: the masked document
    """
    try:
        value = pii_record["value"]
        if value in cache:
            scrambled_ip = cache[value]
        else:
            scrambled_ip = _scramble_ip_address(value)
            cache[value] = scrambled_ip
        document["text"] = _replace_segment(
            document["text"], pii_record["start_pos"], pii_record["end_pos"], scrambled_ip
        )
    except ValueError:
        logger.warning(f'"{value}" not a valid ip address')

    return document


def multilingual_mask_document(
    document: dict[str, Any], pii_records: list[dict[str, Any]], mask_fields: list[str] | None = None
) -> dict[str, Any]:
    """Mask a complete document.

    Mask PII records in a document by replacing them with scrambled values.

    :param document: document to mask
    :param pii_records: PII records to mask
    :param mask_fields: not used! Fields/labels to mask, case-sensitive. None means a
        default list of labels. An empty list means no masking.
    :returns: the masked document
    """
    try:
        cache = {}
        for pii_record in pii_records:
            match pii_record["name"]:
                case (
                    "BANK_ACCOUNT"
                    | "CREDIT_CARD"
                    | "DRIVER_LICENSE"
                    | "GOV_ID"
                    | "LICENSE_PLATE"
                    | "PHONE_NUMBER"
                    | "MERGED"
                ):
                    document = _mask_with_scrambled_string(document, pii_record, cache)
                case "BITCOIN_ADDRESS":
                    document = _mask_bitcoin_address(document, pii_record, cache)
                case "EMAIL_ADDRESS":
                    document = _mask_email_address(document, pii_record, cache)
                case "IP_ADDRESS":
                    document = _mask_ip_address(document, pii_record, cache)
                case _:
                    logger.warning(
                        f"Unknown pii record type {pii_record['name']} in document {document['id']},"
                        f" masked as scrambled string"
                    )
                    document = _mask_with_scrambled_string(document, pii_record, cache)
                    document["pii_unknown"] = True
    except ValueError as e:
        logger.warning(f"Document {document['id']} has pii issues {e}")
    document["pii_masks"] = len(pii_records)
    return document


def openai_mask_document(
    document: dict[str, Any], pii_records: list[dict[str, Any]], mask_fields: list[str] | None = None
) -> dict[str, Any]:
    """Mask a complete document.

    Mask PII records produced by the OpenAIPrivacy tool and wrapper, by replacing them
    with scrambled values.

    :param document: document to mask
    :param pii_records: PII records to mask
    :param mask_fields: fields/labels to mask, case-sensitive. None means a default list
        of labels. An empty list means no masking.
    :returns: the masked document
    """
    try:
        masked_occations = 0
        cache = {}
        if mask_fields is None:
            mask_fields = ["private_email", "private_phone", "private_url", "secret"]
        for pii_record in pii_records:
            if pii_record["name"] not in mask_fields:
                continue
            match pii_record["name"]:
                case (
                    "private_phone"
                    | "secret"
                    | "private_person"
                    | "private_address"
                    | "private_date"
                    | "account_number"
                ):
                    document = _mask_with_scrambled_string(document, pii_record, cache)
                    masked_occations += 1
                case "private_email":
                    document = _mask_email_address(document, pii_record, cache)
                    masked_occations += 1
                case "private_url":
                    if not pii_record["value"].startswith("http"):
                        document = _mask_ip_address(document, pii_record, cache)
                        masked_occations += 1
                case _:
                    logger.warning(
                        f"Unknown pii record type {pii_record['name']} in document {document['id']},"
                        f" masked as scrambled string"
                    )
                    document = _mask_with_scrambled_string(document, pii_record, cache)
                    masked_occations += 1
                    document["pii_unknown"] = True
    except ValueError as e:
        logger.warning(f"Document {document['id']} has pii issues {e}")
    document["pii_masks"] = masked_occations
    document["pii_masks_ignored"] = len(pii_records) - masked_occations
    return document


class PIIMasker:
    """Mask PII records in documents.

    Mask PII records in documents by replacing them with scrambled values.

    :param masker_fn: function that masks a document given its PII records
    :param part_config: configuration of the part, where ``mask`` lists the labels to mask
    :param metric_name: key used for the metrics
    """

    def __init__(
        self,
        masker_fn=multilingual_mask_document,
        part_config: dict[str, Any] | None = None,
        metric_name: str = "pii_masker",
    ) -> None:
        self._metric_name = metric_name
        self._masked_documents = 0
        self._pii_documents = 0
        self._masker_fn = masker_fn
        if part_config is None:
            part_config = {}
        self._mask_fields = part_config.get("mask", None)
        if self._mask_fields is not None:
            logger.info(f"Metadata declare following labels for masking: {', '.join(self._mask_fields)}")

    def get_metrics(self):
        """Get metrics for the masker.

        :returns: dictionary with metrics
        """
        return {
            self._metric_name: {
                "document_masked": self._masked_documents,
                "pii_records": self._pii_documents,
            }
        }

    def get_masker(self, pii_iter: Iterable[dict[str, Any]]) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """Get a mask function.

        Group the PII records by document id, merge overlapping PII records and remove duplicates.

        :param pii_iter: iterator of PII records
        :returns: function that masks a provided document
        """
        pii_grouped = itertools.groupby(pii_iter, lambda x: x["id"])
        # Sort reverse to do the masking from end of doc. That means position of masking is not changed
        # by other masking operation.
        pii_records_sorted = {i: sorted(records, key=lambda x: -x["start_pos"]) for i, records in pii_grouped}
        pii_records_deduped = {
            i: _remove_duplicates_inplace(list(records)) for i, records in pii_records_sorted.items()
        }
        pii = {
            i: _merge_overlapping_ranges(records) if _has_overlapping_ranges(records) else records
            for i, records in pii_records_deduped.items()
        }
        self._pii_documents += len(pii)

        def masker(document):
            if "id" in document and document["id"] in pii:
                self._masked_documents += 1
                return self._masker_fn(document, pii[document["id"]], self._mask_fields)
            else:
                return document

        return masker
