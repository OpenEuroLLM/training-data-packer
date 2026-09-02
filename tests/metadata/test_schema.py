import unittest
from importlib import resources

from parameterized import parameterized

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
        yaml_data = read_metadata(self.resource_path.joinpath(filename))
        metadata = Metadata(yaml_data)
        result, message = self.validator.validate_metadata(metadata)
        self.assertEqual(validates, result, message)

    def test_validator_initialization(self):
        self.assertIsNotNone(self.validator.registry)

    def test_validator_error_message_format(self):
        illegal_data = {"input": "source", "pack": "tree", "sample": "full"}

        success, error = self.validator.validate_metadata(illegal_data)
        self.assertFalse(success)
        self.assertIsNotNone(error)
        self.assertIn("https://https://openeurollm.eu/schemas/metadata.json", error)


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
        success, error = self.validator.validate_release_part(self.complete_record)
        self.assertTrue(success)
        self.assertEqual("", error)

    def test_validator_error_message_format(self):
        illegal_data = {"input": "source", "pack": "tree", "sample": "full"}

        success, error = self.validator.validate_release_part(illegal_data)
        self.assertFalse(success)
        self.assertIsNotNone(error)
        self.assertIn("https://https://openeurollm.eu/schemas/release-part.json", error)

    def test_validate_release_part_missing_required_fields(self):
        del self.complete_record["shard"]

        success, error = self.validator.validate_release_part(self.complete_record)
        self.assertFalse(success)
        self.assertNotEqual("", error)
        self.assertIn("Validation error", error)

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
        success, error = self.validator.validate_release_part(self.complete_record)
        self.assertTrue(success, f"Failed for sample: {sample}\n{error}")

    def test_validate_release_part_invalid_sample_value(self):
        self.complete_record["sample"] = "invalid_sample"
        success, error = self.validator.validate_release_part(self.complete_record)
        self.assertFalse(success)

    @parameterized.expand(
        [
            "tree",
            "flat",
        ]
    )
    def test_validate_release_part_all_pack_values(self, pack):
        self.complete_record["pack"] = pack
        success, error = self.validator.validate_release_part(self.complete_record)
        self.assertTrue(success, f"Failed for sample: {pack}\n{error}")

    def test_validate_release_part_invalid_pack_value(self):
        self.complete_record["pack"] = "invalid_pack"
        success, error = self.validator.validate_release_part(self.complete_record)
        self.assertFalse(success)

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
        success, error = self.validator.validate_release_part(self.complete_record)
        self.assertTrue(success, f"Failed for sample: {mask}\n{error}")

    def test_validate_release_part_invalid_mask_value(self):
        self.complete_record["mask"] = ["invalid_mask"]
        success, error = self.validator.validate_release_part(self.complete_record)
        self.assertFalse(success)

    def test_validate_release_part_mask_empty_list(self):
        self.complete_record["mask"] = []
        success, error = self.validator.validate_release_part(self.complete_record)
        self.assertTrue(success)

    def test_validate_release_part_with_budget(self):
        self.complete_record["budget"] = "25%"
        success, error = self.validator.validate_release_part(self.complete_record)
        self.assertTrue(success)

    @parameterized.expand(
        [
            "100%",
            "25",
        ]
    )
    def test_validate_release_part_with_illegal_budget(self, budget):
        self.complete_record["budget"] = budget
        success, error = self.validator.validate_release_part(self.complete_record)
        self.assertFalse(success)

    def test_validate_release_part_with_filter_and_parameters(self):
        self.complete_record["filter"] = "../filters/custom_filter.py"
        self.complete_record["parameters"] = {"param1": "value1", "param2": 42}
        success, error = self.validator.validate_release_part(self.complete_record)
        self.assertTrue(success)

    def test_validate_release_part_with_scrub(self):
        self.complete_record["scrub"] = ["xml", "md"]
        success, error = self.validator.validate_release_part(self.complete_record)
        self.assertTrue(success, error)


if __name__ == "__main__":
    unittest.main()
