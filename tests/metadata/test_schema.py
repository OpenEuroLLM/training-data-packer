import unittest

from training_data_packer.metadata.schema import Validator


class SchemaTest(unittest.TestCase):
    def setUp(self):
        self.validator = Validator(["metadata.json", "part.json"])

    def test_validator_initialization(self):
        validator = Validator(["metadata.json", "part.json"])
        self.assertIsNotNone(validator.registry)

    def test_validate_release_part_valid(self):
        part_data = {
            "input": "source",
            "annotations": ["nemo-curator"],
            "mask": ["private_email", "EMAIL_ADDRESS", "PHONE_NUMBER"],
            "pack": "tree",
            "sample": "full",
            "shard": "10md",
        }

        success, error = self.validator.validate_release_part(part_data)
        self.assertTrue(success)
        self.assertIsNone(error)

    def test_validate_release_part_missing_required_fields(self):
        part_data = {"input": "source", "annotations": ["nemo-curator"]}

        success, error = self.validator.validate_release_part(part_data)
        self.assertFalse(success)
        self.assertIsNotNone(error)
        self.assertIn("Validation error", error)

    def test_validate_release_part_invalid_sample_value(self):
        part_data = {
            "input": "source",
            "annotations": ["nemo-curator"],
            "mask": ["email"],
            "pack": "tree",
            "sample": "invalid_sample",
            "shard": "10md",
        }

        success, error = self.validator.validate_release_part(part_data)
        self.assertFalse(success)

    def test_validate_release_part_invalid_pack_value(self):
        part_data = {
            "input": "source",
            "annotations": ["nemo-curator"],
            "mask": ["email"],
            "pack": "invalid_pack",
            "sample": "full",
            "shard": "10md",
        }

        success, error = self.validator.validate_release_part(part_data)
        self.assertFalse(success)

    def test_validate_release_part_valid_mask_values(self):
        valid_masks = ["private_email", "EMAIL_ADDRESS", "CREDIT_CARD"]

        for mask in valid_masks:
            part_data = {
                "input": "source",
                "annotations": ["nemo-curator"],
                "mask": [mask],
                "pack": "tree",
                "sample": "full",
                "shard": "10md",
            }

            success, error = self.validator.validate_release_part(part_data)
            self.assertTrue(success, f"Failed for mask: {mask}")

    def test_validate_release_part_invalid_mask_value(self):
        part_data = {
            "input": "source",
            "annotations": ["nemo-curator"],
            "mask": ["invalid_mask"],
            "pack": "tree",
            "sample": "full",
            "shard": "10md",
        }

        success, error = self.validator.validate_release_part(part_data)
        self.assertFalse(success)

    def test_validate_release_part_all_sample_values(self):
        valid_samples = ["full", "dynamic", "random", "wds+register"]

        for sample in valid_samples:
            part_data = {
                "input": "source",
                "annotations": ["nemo-curator"],
                "mask": [],
                "pack": "tree",
                "sample": sample,
                "shard": "10md",
            }

            success, error = self.validator.validate_release_part(part_data)
            self.assertTrue(success, f"Failed for sample: {sample}")

    def test_validate_release_part_both_pack_values(self):
        valid_packs = ["flat", "tree"]

        for pack in valid_packs:
            part_data = {
                "input": "source",
                "annotations": ["nemo-curator"],
                "mask": [],
                "pack": pack,
                "sample": "full",
                "shard": "10md",
            }

            success, error = self.validator.validate_release_part(part_data)
            self.assertTrue(success, f"Failed for pack: {pack}, error: {error}")

    def test_validate_release_part_empty_optional_fields(self):
        part_data = {
            "input": "source",
            "annotations": ["nemo-curator"],
            "mask": [],
            "pack": "tree",
            "sample": "full",
            "shard": "10md",
        }

        success, error = self.validator.validate_release_part(part_data)
        self.assertTrue(success)

    def test_validate_release_part_with_budget(self):
        part_data = {
            "input": "source",
            "annotations": ["nemo-curator"],
            "mask": [],
            "pack": "tree",
            "sample": "random",
            "shard": "10md",
            "budget": "25%",
        }

        success, error = self.validator.validate_release_part(part_data)
        self.assertTrue(success)

    def test_validate_release_part_with_filter(self):
        part_data = {
            "input": "source",
            "annotations": ["nemo-curator"],
            "mask": [],
            "pack": "tree",
            "sample": "dynamic",
            "shard": "5md",
            "filter": "../filters/custom_filter.py",
        }

        success, error = self.validator.validate_release_part(part_data)
        self.assertTrue(success)

    def test_validate_release_part_with_scrub(self):
        part_data = {
            "input": "source",
            "annotations": ["nemo-curator"],
            "mask": [],
            "pack": "tree",
            "sample": "full",
            "shard": "10md",
            "scrub": ["field1", "field2"],
        }

        success, error = self.validator.validate_release_part(part_data)
        self.assertTrue(success)

    def test_validate_release_part_with_parameters(self):
        part_data = {
            "input": "source",
            "annotations": ["nemo-curator"],
            "mask": [],
            "pack": "tree",
            "sample": "dynamic",
            "shard": "10md",
            "parameters": {"param1": "value1", "param2": 42},
        }

        success, error = self.validator.validate_release_part(part_data)
        self.assertTrue(success)

    def test_validator_with_single_schema(self):
        validator = Validator(["part.json"])
        part_data = {
            "input": "source",
            "annotations": ["nemo-curator"],
            "mask": [],
            "pack": "flat",
            "sample": "full",
            "shard": "15bd",
        }

        success, error = validator.validate_release_part(part_data)
        self.assertTrue(success)

    def test_validator_error_message_format(self):
        part_data = {"input": "source", "pack": "tree", "sample": "full"}
        # Missing required fields: annotations, mask, shard

        success, error = self.validator.validate_release_part(part_data)
        self.assertFalse(success)
        self.assertIsNotNone(error)
        self.assertIn("https://https://openeurollm.eu/schemas/part.json", error)


if __name__ == "__main__":
    unittest.main()
