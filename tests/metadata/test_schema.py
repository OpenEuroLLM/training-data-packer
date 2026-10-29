import copy
import unittest
from importlib import resources

import pytest
from parameterized import parameterized

import tests.helpers.metadata
import tests.resources.metadata
from training_data_packer.metadata import Metadata, read_metadata
from training_data_packer.metadata.schema import Validator


class MetadataSchemaTest(unittest.TestCase):
    def setUp(self):
        self.validator = Validator()
        self.resource_path = resources.files(tests.resources.metadata)

    @parameterized.expand(
        [
            ["full file", "common-pile.yaml", True],
            ["not-allowed-prop-release-default", "not-allowed-prop-release-default.yaml", False],
            ["everything", "everything.yaml", True],
        ]
    )
    def test_validate_real_file(self, name: str, filename: str, validates: bool):
        if validates:
            read_metadata(self.resource_path.joinpath(filename))  # Raises exception if validation fails.
        else:
            with pytest.raises(ValueError):
                read_metadata(self.resource_path.joinpath(filename))

    def test_validator_initialization(self):
        self.assertIsNotNone(self.validator.registry)


class MegatronLmSchemaTest(unittest.TestCase):
    def setUp(self):
        self.validator = Validator()
        self.metadata = Metadata(copy.deepcopy(tests.helpers.metadata.minimal_metadata_dict))

    @parameterized.expand(
        [
            ["minimal", {"default": {"source": "release"}, "tok-256k": {"huggingface": "org/tokenizer-256k"}}],
            [
                "several tokenizers",
                {
                    "default": {"source": "release"},
                    "tok-256k": {"huggingface": "org/tokenizer-256k"},
                    "tok_v2.1": {"huggingface": "/path/to/tokenizer"},
                },
            ],
            [
                "bos and eos",
                {
                    "default": {"source": "release", "bos": False, "eos": "<|endoftext|>"},
                    "tok": {"huggingface": "org/tok", "bos": "<s>", "eos": False},
                },
            ],
            [
                "source override",
                {"default": {"source": "release"}, "tok": {"huggingface": "org/tok", "source": "release-raw"}},
            ],
        ]
    )
    def test_valid(self, _name: str, megatron_lm: dict):
        self.metadata["megatron-lm"] = megatron_lm
        self.assertTrue(self.validator.validate_metadata(self.metadata))

    @parameterized.expand(
        [
            ["no tokenizer", {"default": {"source": "release"}}],
            ["missing default", {"tok": {"huggingface": "org/tok"}}],
            ["missing default source", {"default": {}, "tok": {"huggingface": "org/tok"}}],
            ["unknown default field", {"default": {"source": "release", "foo": 1}, "tok": {"huggingface": "org/tok"}}],
            ["missing huggingface", {"default": {"source": "release"}, "tok": {}}],
            ["null tokenizer", {"default": {"source": "release"}, "tok": None}],
            ["empty huggingface", {"default": {"source": "release"}, "tok": {"huggingface": ""}}],
            ["unknown tokenizer field", {"default": {"source": "release"}, "tok": {"huggingface": "o/t", "foo": 1}}],
            ["empty bos", {"default": {"source": "release", "bos": ""}, "tok": {"huggingface": "org/tok"}}],
            ["true eos", {"default": {"source": "release"}, "tok": {"huggingface": "org/tok", "eos": True}}],
            ["slash in name", {"default": {"source": "release"}, "org/tok": {"huggingface": "org/tok"}}],
        ]
    )
    def test_invalid(self, _name: str, megatron_lm: dict):
        self.metadata["megatron-lm"] = megatron_lm
        with pytest.raises(ValueError):
            self.validator.validate_metadata(self.metadata)

    def test_missing_section_valid(self):
        del self.metadata.data["megatron-lm"]
        self.assertTrue(self.validator.validate_metadata(self.metadata))

    def test_tokenizer_field_not_allowed(self):
        self.metadata.data["tokenizer"] = "org/tok"
        with pytest.raises(ValueError):
            self.validator.validate_metadata(self.metadata)


class ReleasePartSchemaTest(unittest.TestCase):
    def setUp(self):
        self.validator = Validator()
        self.complete_record = {
            "input": "source",
            "annotations": ["nemo-curator"],
            "mask": ["private_email", "EMAIL_ADDRESS", "PHONE_NUMBER"],
            "pack": "tree",
            "sample": "full",
            "shard": "10md",
        }

    def test_validate_release_part_valid(self):
        success = self.validator.validate_release_part(self.complete_record)
        self.assertTrue(success)

    def test_validate_release_part_missing_required_fields(self):
        del self.complete_record["shard"]

        with pytest.raises(ValueError):
            self.validator.validate_release_part(self.complete_record)

    def test_validate_release_part_error_message_is_single_line(self):
        del self.complete_record["shard"]

        with pytest.raises(ValueError) as exc_info:
            self.validator.validate_release_part(self.complete_record)
        message = str(exc_info.value)
        self.assertNotIn("\n", message)
        self.assertIn("'shard' is a required property", message)

    @parameterized.expand(
        [
            "full",
            "dynamic",
            "random",
            "wds+register",
        ]
    )
    def test_validate_release_part_all_sample_values(self, sample):
        self.complete_record["sample"] = sample
        success = self.validator.validate_release_part(self.complete_record)
        self.assertTrue(success)

    def test_validate_release_part_invalid_sample_value(self):
        self.complete_record["sample"] = "invalid_sample"
        with pytest.raises(ValueError):
            self.validator.validate_release_part(self.complete_record)

    @parameterized.expand(
        [
            "tree",
            "flat",
        ]
    )
    def test_validate_release_part_all_pack_values(self, pack):
        self.complete_record["pack"] = pack
        success = self.validator.validate_release_part(self.complete_record)
        self.assertTrue(success)

    def test_validate_release_part_invalid_pack_value(self):
        self.complete_record["pack"] = "invalid_pack"
        with pytest.raises(ValueError):
            self.validator.validate_release_part(self.complete_record)

    @parameterized.expand(
        [
            "account_number",
            "private_address",
            "private_date",
            "private_email",
            "private_person",
            "private_phone",
            "private_url",
            "secret",
            "BANK_ACCOUNT",
            "BITCOIN_ADDRESS",
            "CREDIT_CARD",
            "DRIVER_LICENSE",
            "EMAIL_ADDRESS",
            "GOV_ID",
            "IP_ADDRESS",
            "LICENSE_PLATE",
            "PHONE_NUMBER",
        ]
    )
    def test_validate_release_part_all_mask_values(self, mask):
        self.complete_record["mask"] = [mask]
        success = self.validator.validate_release_part(self.complete_record)
        self.assertTrue(success)

    def test_validate_release_part_invalid_mask_value(self):
        self.complete_record["mask"] = ["invalid_mask"]
        with pytest.raises(ValueError):
            self.validator.validate_release_part(self.complete_record)

    def test_validate_release_part_mask_empty_list(self):
        self.complete_record["mask"] = []
        success = self.validator.validate_release_part(self.complete_record)
        self.assertTrue(success)

    def test_validate_release_part_with_budget(self):
        self.complete_record["budget"] = "25%"
        success = self.validator.validate_release_part(self.complete_record)
        self.assertTrue(success)

    @parameterized.expand(
        [
            "100%",
            "25",
        ]
    )
    def test_validate_release_part_with_illegal_budget(self, budget):
        self.complete_record["budget"] = budget
        with pytest.raises(ValueError):
            self.validator.validate_release_part(self.complete_record)

    def test_validate_release_part_with_filter_and_parameters(self):
        self.complete_record["filter"] = "../filters/custom_filter.py"
        self.complete_record["parameters"] = {"param1": "value1", "param2": 42}
        success = self.validator.validate_release_part(self.complete_record)
        self.assertTrue(success)

    def test_validate_release_part_with_scrub(self):
        self.complete_record["scrub"] = ["xml", "md"]
        success = self.validator.validate_release_part(self.complete_record)
        self.assertTrue(success)


if __name__ == "__main__":
    unittest.main()
