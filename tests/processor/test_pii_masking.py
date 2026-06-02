import copy
import ipaddress
import unittest

from parameterized import parameterized

from training_data_packer.processor import pii_masking


class TestPiiUtilityFunctions(unittest.TestCase):
    @parameterized.expand(
        [
            ["no_pii", [], False],
            ["single_pii", [{"start_pos": 1, "end_pos": 5}], False],
            [
                "no_overlap",
                [
                    {"start_pos": 12, "end_pos": 15},
                    {"start_pos": 6, "end_pos": 8},
                    {"start_pos": 1, "end_pos": 5},
                ],
                False,
            ],
            [
                "overlap_1",
                [
                    {"start_pos": 12, "end_pos": 15},
                    {"start_pos": 4, "end_pos": 8},
                    {"start_pos": 1, "end_pos": 5},
                ],
                True,
            ],
            [
                "overlap_2",
                [
                    {"start_pos": 8, "end_pos": 15},
                    {"start_pos": 6, "end_pos": 8},
                    {"start_pos": 1, "end_pos": 7},
                ],
                True,
            ],
        ]
    )
    def test_has_overlapping_ranges(self, name, pii_records, expected):
        self.assertEqual(pii_masking._has_overlapping_ranges(pii_records), expected)

    @parameterized.expand(
        [
            ["no_pii", [], [], set()],
            [
                "single_pii",
                [{"start_pos": 1, "end_pos": 6, "value": "12345", "name": "A"}],
                [{"start_pos": 1, "end_pos": 6, "value": "12345", "name": "A"}],
                set(),
            ],
            [
                "no_overlap",
                [
                    {"start_pos": 12, "end_pos": 16, "value": "2345", "name": "A"},
                    {"start_pos": 6, "end_pos": 9, "value": "678", "name": "A"},
                    {"start_pos": 1, "end_pos": 6, "value": "12345", "name": "A"},
                ],
                [
                    {"start_pos": 12, "end_pos": 16, "value": "2345", "name": "A"},
                    {"start_pos": 6, "end_pos": 9, "value": "678", "name": "A"},
                    {"start_pos": 1, "end_pos": 6, "value": "12345", "name": "A"},
                ],
                set(),
            ],
            [
                "overlap_1",
                [
                    {"start_pos": 12, "end_pos": 16, "value": "2345", "name": "A"},
                    {"start_pos": 4, "end_pos": 9, "value": "45678", "name": "B"},
                    {"start_pos": 1, "end_pos": 6, "value": "12345", "name": "C"},
                ],
                [
                    {"start_pos": 12, "end_pos": 16, "value": "2345", "name": "A"},
                    {"start_pos": 1, "end_pos": 9, "value": "12345678", "name": "MERGED"},
                ],
                {"B", "C"},
            ],
            [
                "overlap_2",
                [
                    {"start_pos": 8, "end_pos": 16, "value": "89012345", "name": "A"},
                    {"start_pos": 6, "end_pos": 9, "value": "678", "name": "B"},
                    {"start_pos": 1, "end_pos": 6, "value": "12345", "name": "C"},
                ],
                [
                    {
                        "start_pos": 6,
                        "end_pos": 16,
                        "value": "6789012345",
                        "name": "MERGED",
                    },
                    {"start_pos": 1, "end_pos": 6, "value": "12345", "name": "C"},
                ],
                {"A", "B"},
            ],
            [
                "Real case",
                [
                    {
                        "WARC-Record-ID": "<urn:uuid:1d45dcd5-aeaa-4527-a97f-adda7d995954>",
                        "name": "PHONE_NUMBER",
                        "value": "337-463-4486",
                        "start_pos": 2469,
                        "end_pos": 2481,
                    },
                    {
                        "WARC-Record-ID": "<urn:uuid:1d45dcd5-aeaa-4527-a97f-adda7d995954>",
                        "name": "GOV_ID",
                        "value": "634 337-463",
                        "start_pos": 2465,
                        "end_pos": 2476,
                    },
                ],
                [
                    {
                        "WARC-Record-ID": "<urn:uuid:1d45dcd5-aeaa-4527-a97f-adda7d995954>",
                        "name": "MERGED",
                        "value": "634 337-463-4486",
                        "start_pos": 2465,
                        "end_pos": 2481,
                    }
                ],
                {"GOV_ID", "PHONE_NUMBER"},
            ],
        ]
    )
    def test_merge_overlapping_ranges(self, name, pii_records, expected, expected_merged):
        result = pii_masking._merge_overlapping_ranges(pii_records)
        self.assertEqual(result, expected)

    @parameterized.expand(
        [
            ["no_pii", [], []],
            [
                "single_pii",
                [{"start_pos": 1, "end_pos": 6, "value": "12345"}],
                [{"start_pos": 1, "end_pos": 6, "value": "12345"}],
            ],
            [
                "duplicate",
                [
                    {"start_pos": 6, "end_pos": 9, "value": "678"},
                    {"start_pos": 6, "end_pos": 9, "value": "678"},
                    {"start_pos": 1, "end_pos": 6, "value": "12345"},
                ],
                [
                    {"start_pos": 6, "end_pos": 9, "value": "678"},
                    {"start_pos": 1, "end_pos": 6, "value": "12345"},
                ],
            ],
        ]
    )
    def test_remove_duplicates_inplace(self, name, pii_records, expected):
        self.assertEqual(pii_masking._remove_duplicates_inplace(pii_records), expected)

    @parameterized.expand(
        [
            ["happy_path", 5, 9, "XXXXXXX", "Some XXXXXXX to replace"],
            ["pii_at_start", 0, 2, "X", "Xme text to replace"],
            ["pii_at_end", 18, 20, "X", "Some text to replaX"],
            ["pii_at_single_position", 5, 6, "X", "Some Xext to replace"],
        ]
    )
    def test_replace_segment(self, name, start_pos, end_pos, replacement, expected):
        text = "Some text to replace"
        result = pii_masking._replace_segment(text, start_pos, end_pos, replacement)
        self.assertEqual(result, expected)

    @parameterized.expand(
        [
            ["pii_start_after", 18, 100],
            ["pii_at_end_and_after", 3, 100],
            ["pii_start_at_same_as_length", 5, 5],
            ["pii_end_at_same_as_length", 3, 6],
        ]
    )
    def test_replace_after_text(self, name, start_pos, end_pos):
        with self.assertRaises(ValueError):
            pii_masking._replace_segment("01234", start_pos, end_pos, "XX")

    @parameterized.expand(
        [
            ["phone-number", "+1 (505) 619 5504", "EDEEDDDEEDDDEDDDD"],
        ]
    )
    def test_scramble_text(self, name, input, string_format):
        result = pii_masking._scramble_string(input)
        self.assertEqual(len(input), len(result))
        self.assertNotEqual(input, result)
        for k, c in enumerate(string_format):
            match c:
                case "E":
                    self.assertEqual(input[k], result[k])
                case "D":
                    self.assertTrue(result[k].isdigit())
                case "L":
                    self.assertTrue(result[k].isalpha())
                    self.assertEqual(result[k].islower(), input[k].islower())
                case _:
                    self.fail(f"{input} was scrambled to {result} which is not in the format {string_format}")

    @parameterized.expand(
        [
            ["well_known_dns_ipv4", "8.8.8.8", 4],
            ["link_local_ipv4", "169.254.169.254", 4],
            ["well_known_dns_ipv6", "2001:db8::1000", 6],
            ["link_local_ipv6", "fe80::1234", 6],
        ]
    )
    def test_scramble_ip_address(self, name, ip_address, version):
        result = pii_masking._scramble_ip_address(ip_address)
        result_ip = ipaddress.ip_address(result)
        self.assertEqual(version, result_ip.version)
        self.assertNotEqual(ip_address, result)

    @parameterized.expand([["localhost_ipv4", "127.0.0.1"], ["localhost_ipv6", "::1"]])
    def test_not_scramble_localhost(self, name, ip_address):
        result = pii_masking._scramble_ip_address(ip_address)
        self.assertEqual(ip_address, result)

    def test_scramble_illegal_ip(self):
        with self.assertRaises(ValueError):
            pii_masking._scramble_ip_address("300.300.300.300")


class TestMaskRecords(unittest.TestCase):
    @parameterized.expand(
        [
            [
                "happy_path",
                "This is my email pii@example.org.",
                {"start_pos": 17, "end_pos": 32},
                "This is my email test@example.com.",
            ],
        ]
    )
    def test_mask_email_address(self, name, text, pii_record, expected):
        document = {"text": text}
        result = pii_masking._mask_email_address(document, pii_record)
        self.assertEqual(result["text"], expected)

    @parameterized.expand(
        [
            [
                "happy_path_ipv4",
                "The IP 2001:db8::1000 is a public DNS.",
                {"start_pos": 7, "end_pos": 20, "value": "2001:db8::1000"},
            ],
            [
                "happy_path_ipv6",
                "The IP 2001:db8::1000 is a public DNS.",
                {"start_pos": 7, "end_pos": 20, "value": "2001:db8::1000"},
            ],
            [
                "faulty_ip",
                "The IP 300.300.300.300 is a public DNS.",
                {"start_pos": 7, "end_pos": 20, "value": "300.300.300.300"},
            ],
        ]
    )
    def test_mask_ip_address(self, name, text, pii_record):
        document = {"text": text}
        result = pii_masking._mask_ip_address(document, pii_record)
        self.assertEqual(result["text"][: pii_record["start_pos"]], text[: pii_record["start_pos"]])
        self.assertNotEqual(
            result["text"][pii_record["start_pos"] : pii_record["end_pos"]],
            text[pii_record["start_pos"] : pii_record["end_pos"]],
        )
        suffix_text = text[pii_record["end_pos"] + 1 :]
        self.assertEqual(result["text"][-len(suffix_text) :], suffix_text)

    @parameterized.expand(
        [
            [
                "happy_path",
                "This is my email +1 (505) 619 5504.",
                {"start_pos": 17, "end_pos": 34, "value": "+1 (505) 619 5504"},
            ],
        ]
    )
    def test_mask_with_scrambled_string(self, name, text, pii_record):
        document = {"text": text}
        result = pii_masking._mask_with_scrambled_string(document, pii_record)
        self.assertEqual(result["text"][: pii_record["start_pos"]], text[: pii_record["start_pos"]])
        self.assertNotEqual(
            result["text"][pii_record["start_pos"] : pii_record["end_pos"]],
            text[pii_record["start_pos"] : pii_record["end_pos"]],
        )
        self.assertEqual(result["text"][pii_record["end_pos"] + 1 :], text[pii_record["end_pos"] + 1 :])

    @parameterized.expand(
        [
            [
                "bitcoin_with_prefix_1",
                "This is my bitcoin 1BgXRx8YMGKU7fc8RfTPQ2uL2ivC9cMmGj give me the money",
                {
                    "start_pos": 19,
                    "end_pos": 53,
                    "value": "1BgXRx8YMGKU7fc8RfTPQ2uL2ivC9cMmGj",
                    "id": "1",
                },
                1,
            ],
            [
                "bitcoin_with_prefix_3",
                "This is my bitcoin 3G3CxtfN4rg4ShVzVAUsM3AtGbnbs6v26S give me the money",
                {
                    "start_pos": 19,
                    "end_pos": 53,
                    "value": "3G3CxtfN4rg4ShVzVAUsM3AtGbnbs6v26S",
                    "id": "2",
                },
                1,
            ],
            [
                "bitcoin_with_prefix_bc1",
                "This is my bitcoin bc1qsfc2g86agexaht5eh8yfp3xs9q65nhnz85dq8u give me the money",
                {
                    "start_pos": 19,
                    "end_pos": 61,
                    "value": "bc1qsfc2g86agexaht5eh8yfp3xs9q65nhnz85dq8u",
                    "id": "3",
                },
                3,
            ],
            [
                "bitcoin_that_is_invalid",
                "This is my bitcoin 7BgXRx8YMGKU7fc8RfTPQ2uL2ivC9cMmGj give me the money",
                {
                    "start_pos": 19,
                    "end_pos": 53,
                    "value": "7BgXRx8YMGKU7fc8RfTPQ2uL2ivC9cMmGj",
                    "id": "4",
                },
                0,
            ],
        ]
    )
    def test_mask_bitcoin_address(self, name, text, pii_record, prefix_length):
        document = {"text": text}
        result = pii_masking._mask_bitcoin_address(document, pii_record)
        self.assertEqual(
            result["text"][: pii_record["start_pos"] + prefix_length],
            text[: pii_record["start_pos"] + prefix_length],
        )
        self.assertNotEqual(
            result["text"][pii_record["start_pos"] + prefix_length : pii_record["end_pos"]],
            text[pii_record["start_pos"] + prefix_length : pii_record["end_pos"]],
        )
        self.assertEqual(result["text"][pii_record["end_pos"] + 1 :], text[pii_record["end_pos"] + 1 :])

    @parameterized.expand(
        [
            ["no_masking", "This is a random text.", [], 0, False],
            [
                "mask_bank_account",
                "523 789This is a random text.",
                [
                    {
                        "start_pos": 0,
                        "end_pos": 6,
                        "value": "523 789",
                        "id": "2",
                        "name": "BANK_ACCOUNT",
                    },
                ],
                1,
                False,
            ],
            [
                "mask_credit_card",
                "5234 7894 This is a random text.",
                [
                    {
                        "start_pos": 0,
                        "end_pos": 8,
                        "value": "5234 7894",
                        "id": "2",
                        "name": "CREDIT_CARD",
                    },
                ],
                1,
                False,
            ],
            [
                "mask_driver_license",
                "52347894 This is a random text.",
                [
                    {
                        "start_pos": 0,
                        "end_pos": 7,
                        "value": "52347894",
                        "id": "2",
                        "name": "DRIVER_LICENSE",
                    },
                ],
                1,
                False,
            ],
            [
                "mask_unknown",
                "52347894 This is a random text.",
                [
                    {
                        "start_pos": 0,
                        "end_pos": 7,
                        "value": "52347894",
                        "id": "2",
                        "name": "UNKNOWN",
                    },
                ],
                1,
                True,
            ],
        ]
    )
    def test_mask_document_has_changed_docs(self, name, text, pii_records, expected, unknown_pii):
        masked_doc = pii_masking.mask_document({"id": "1234", "text": text}, pii_records)
        self.assertEqual(expected, masked_doc["pii_masks"])
        if unknown_pii:
            self.assertTrue(masked_doc["pii_unknown"])
        else:
            self.assertFalse("pii_unknown" in masked_doc)
        if masked_doc["pii_masks"] > 0:
            self.assertNotEqual(masked_doc["text"], text)
        else:
            self.assertEqual(masked_doc["text"], text)


class TestPIIMasker(unittest.TestCase):
    def test_happy_path(self):
        src_indata = [
            {"id": "1234", "text": "This is a random text."},
            {"id": "1235", "text": "5234 7894 This is a random text. foo@example.org"},
            {"id": "1236", "text": "This is a random text."},
        ]
        pii_indata = [
            {
                "start_pos": 0,
                "end_pos": 8,
                "value": "5234 7894",
                "id": "1235",
                "name": "CREDIT_CARD",
            },
            {
                "start_pos": 33,
                "end_pos": 47,
                "value": "foo@example.org",
                "id": "1235",
                "name": "EMAIL_ADDRESS",
            },
        ]
        src_indata_iter = iter(copy.deepcopy(src_indata))
        pii_indata_iter = iter(pii_indata)
        pii_masker = pii_masking.PIIMasker()
        pii_list = list(map(pii_masker.get_masker(pii_indata_iter), src_indata_iter))
        self.assertEqual(pii_list[0], src_indata[0])
        self.assertNotEqual(pii_list[1], src_indata[1])
        self.assertEqual(pii_list[1]["pii_masks"], 2)
        self.assertEqual(pii_list[2], src_indata[2])
        self.assertEqual(list(pii_indata_iter), [])
        self.assertEqual(
            pii_masker.get_metrics(),
            {
                "pii_masker": {
                    "masked_documents": 1,
                    "pii_documents": 1,
                }
            },
        )


if __name__ == "__main__":
    unittest.main()
