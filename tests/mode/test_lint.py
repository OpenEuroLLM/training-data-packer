import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training_data_packer.mode.lint import _check_pack_config, _check_parts_and_dirs_match, _check_sample_config


class TestCheckPartsAndDirsMatch(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_matching_directories_returns_true(self):
        directory = Path(self.temp_dir)
        part_dirs = ["part1", "part2", "part3"]
        for part_dir in part_dirs:
            (directory / part_dir).mkdir()

        result = _check_parts_and_dirs_match(directory, part_dirs)
        self.assertTrue(result)

    def test_extra_directory_returns_false(self):
        directory = Path(self.temp_dir)
        part_names = ["part1", "part2"]
        actual_dirs = ["part1", "part2", "extra_dir"]
        for dir_name in actual_dirs:
            (directory / dir_name).mkdir()

        with patch("training_data_packer.mode.lint.logger") as mock_logger:
            result = _check_parts_and_dirs_match(directory, part_names)
            self.assertFalse(result)
            mock_logger.warning.assert_called_once()
            self.assertIn("extra_dir", str(mock_logger.warning.call_args))

    def test_missing_part_directory_returns_false(self):
        directory = Path(self.temp_dir)
        part_names = ["part1", "part2", "part3"]
        for dir_name in ["part1", "part2"]:
            (directory / dir_name).mkdir()

        with patch("training_data_packer.mode.lint.logger") as mock_logger:
            result = _check_parts_and_dirs_match(directory, part_names)
            self.assertFalse(result)
            mock_logger.warning.assert_called_once()
            self.assertIn("part3", str(mock_logger.warning.call_args))

    def test_empty_directory_returns_false(self):
        directory = Path(self.temp_dir)
        part_names = ["part1", "part2"]

        with patch("training_data_packer.mode.lint.logger") as mock_logger:
            result = _check_parts_and_dirs_match(directory, part_names)
            self.assertFalse(result)
            mock_logger.warning.assert_called_once()
            self.assertIn("part1", str(mock_logger.warning.call_args))

    def test_only_extra_directories_returns_false(self):
        directory = Path(self.temp_dir)
        part_names = []
        actual_dirs = ["extra1", "extra2"]
        for dir_name in actual_dirs:
            (directory / dir_name).mkdir()

        with patch("training_data_packer.mode.lint.logger") as mock_logger:
            result = _check_parts_and_dirs_match(directory, part_names)
            self.assertFalse(result)
            mock_logger.warning.assert_called_once()

    def test_multiple_extra_directories_returns_false(self):
        directory = Path(self.temp_dir)
        part_names = ["part1"]
        actual_dirs = ["part1", "extra1", "extra2", "extra3"]
        for dir_name in actual_dirs:
            (directory / dir_name).mkdir()

        with patch("training_data_packer.mode.lint.logger") as mock_logger:
            result = _check_parts_and_dirs_match(directory, part_names)
            self.assertFalse(result)
            warning_message = str(mock_logger.warning.call_args)
            self.assertIn("extra1", warning_message)
            self.assertIn("extra2", warning_message)
            self.assertIn("extra3", warning_message)

    def test_nested_files_ignored(self):
        directory = Path(self.temp_dir)
        part_names = ["part1"]
        (directory / "part1").mkdir()
        (directory / "part1" / "file.txt").touch()

        result = _check_parts_and_dirs_match(directory, part_names)
        self.assertTrue(result)

    def test_nested_part_directories_returns_true(self):
        directory = Path(self.temp_dir)
        part_names = ["foo/bar", "baz/qux"]
        for part_name in part_names:
            (directory / part_name).mkdir(parents=True)

        result = _check_parts_and_dirs_match(directory, part_names)
        self.assertTrue(result)

    def test_nested_extra_directory_returns_false(self):
        directory = Path(self.temp_dir)
        part_names = ["foo/bar"]
        (directory / "foo" / "bar").mkdir(parents=True)
        (directory / "extra_dir").mkdir()

        with patch("training_data_packer.mode.lint.logger") as mock_logger:
            result = _check_parts_and_dirs_match(directory, part_names)
            self.assertFalse(result)
            mock_logger.warning.assert_called_once()
            self.assertIn("extra_dir", str(mock_logger.warning.call_args))

    def test_missing_nested_part_directory_returns_false(self):
        directory = Path(self.temp_dir)
        part_names = ["foo/bar", "missing/nested"]
        (directory / "foo" / "bar").mkdir(parents=True)

        with patch("training_data_packer.mode.lint.logger") as mock_logger:
            result = _check_parts_and_dirs_match(directory, part_names)
            self.assertFalse(result)
            mock_logger.warning.assert_called_once()
            self.assertIn("missing/nested", str(mock_logger.warning.call_args))

    def test_mixed_flat_and_nested_parts(self):
        directory = Path(self.temp_dir)
        part_names = ["part1", "foo/bar", "baz/qux/deep"]
        for part_name in part_names:
            (directory / part_name).mkdir(parents=True)

        result = _check_parts_and_dirs_match(directory, part_names)
        self.assertTrue(result)

    def test_nested_extra_directory_under_part_returns_false(self):
        directory = Path(self.temp_dir)
        part_names = ["foo/bar"]
        (directory / "foo" / "bar").mkdir(parents=True)
        (directory / "foo" / "extra").mkdir(parents=True)

        with patch("training_data_packer.mode.lint.logger") as mock_logger:
            result = _check_parts_and_dirs_match(directory, part_names)
            self.assertFalse(result)
            warning_message = str(mock_logger.warning.call_args)
            self.assertIn("foo/extra", warning_message)


class TestCheckSampleConfig(unittest.TestCase):
    def setUp(self):
        self.part_path = "release.test-part"

    def test_full_mode_passes(self):
        part_conf = {"sample": "full"}

        self.assertTrue(_check_sample_config(self.part_path, part_conf))

    def test_full_mode_with_budget_raises_error(self):
        part_conf = {"sample": "full", "budget": 1000}

        with self.assertRaises(ValueError) as context:
            _check_sample_config(self.part_path, part_conf)
        self.assertIn("full", str(context.exception))
        self.assertIn("budget", str(context.exception))

    def test_full_mode_with_rubber_raises_error(self):
        part_conf = {"sample": "full", "rubber": 42}

        with self.assertRaises(ValueError) as context:
            _check_sample_config(self.part_path, part_conf)
        self.assertIn("full", str(context.exception))
        self.assertIn("rubber", str(context.exception))

    def test_full_mode_with_filter_raises_error(self):
        part_conf = {"sample": "full", "filter": Path("tests/resources/sampler/test_sampler.py")}

        with self.assertRaises(ValueError) as context:
            _check_sample_config(self.part_path, part_conf)
        self.assertIn("full", str(context.exception))
        self.assertIn("filter", str(context.exception))

    def test_full_mode_with_parameters_raises_error(self):
        part_conf = {"sample": "full", "parameters": {"sample_ratio": "0.5"}}

        with self.assertRaises(ValueError) as context:
            _check_sample_config(self.part_path, part_conf)
        self.assertIn("full", str(context.exception))
        self.assertIn("parameters", str(context.exception))

    def test_full_mode_with_multiple_invalid_fields_raises_error(self):
        part_conf = {
            "sample": "full",
            "budget": 1000,
            "rubber": 42,
            "filter": Path("tests/resources/sampler/test_sampler.py"),
        }

        with self.assertRaises(ValueError) as context:
            _check_sample_config(self.part_path, part_conf)
        error_message = str(context.exception)
        self.assertIn("full", error_message)
        self.assertIn("budget", error_message)
        self.assertIn("rubber", error_message)
        self.assertIn("filter", error_message)

    def test_wds_register_mode_passes(self):
        part_conf = {"sample": "wds+register"}

        self.assertTrue(_check_sample_config(self.part_path, part_conf))

    def test_wds_register_mode_with_budget_raises_error(self):
        part_conf = {"sample": "wds+register", "budget": 1000}

        with self.assertRaises(ValueError) as context:
            _check_sample_config(self.part_path, part_conf)
        self.assertIn("wds+register", str(context.exception))
        self.assertIn("budget", str(context.exception))

    def test_wds_register_mode_with_rubber_raises_error(self):
        part_conf = {"sample": "wds+register", "rubber": 42}

        with self.assertRaises(ValueError) as context:
            _check_sample_config(self.part_path, part_conf)
        self.assertIn("wds+register", str(context.exception))
        self.assertIn("rubber", str(context.exception))

    def test_wds_register_mode_with_filter_raises_error(self):
        part_conf = {"sample": "wds+register", "filter": Path("tests/resources/sampler/test_sampler.py")}

        with self.assertRaises(ValueError) as context:
            _check_sample_config(self.part_path, part_conf)
        self.assertIn("wds+register", str(context.exception))
        self.assertIn("filter", str(context.exception))

    def test_wds_register_mode_with_parameters_raises_error(self):
        part_conf = {"sample": "wds+register", "parameters": {"sample_ratio": "0.5"}}

        with self.assertRaises(ValueError) as context:
            _check_sample_config(self.part_path, part_conf)
        self.assertIn("wds+register", str(context.exception))
        self.assertIn("parameters", str(context.exception))

    def test_wds_register_mode_with_other_allowed_fields_passes(self):
        part_conf = {
            "sample": "wds+register",
            "pack": "flat",
            "prefix": "test",
            "suffix": ".jsonl",
        }

        self.assertTrue(_check_sample_config(self.part_path, part_conf))

    def test_full_mode_with_other_allowed_fields_passes(self):
        part_conf = {
            "sample": "full",
            "pack": "tree",
            "suffix": ".jsonl.gz",
        }

        self.assertTrue(_check_sample_config(self.part_path, part_conf))

    def test_random_mode_without_budget_raises_error(self):
        part_conf = {"sample": "random"}

        with patch("training_data_packer.mode.lint.logger") as mock_logger:
            with self.assertRaises(ValueError) as context:
                _check_sample_config(self.part_path, part_conf)
            self.assertIn("random", str(context.exception))
            self.assertIn("budget", str(context.exception))
            self.assertIn(self.part_path, str(context.exception))
            mock_logger.warning.assert_called_once()

    def test_random_mode_with_budget_passes(self):
        part_conf = {"sample": "random", "budget": 1000}

        self.assertTrue(_check_sample_config(self.part_path, part_conf))

    def test_random_mode_without_rubber_warns(self):
        part_conf = {"sample": "random", "budget": 1000}

        with patch("training_data_packer.mode.lint.logger") as mock_logger:
            _check_sample_config(self.part_path, part_conf)
            mock_logger.warning.assert_called_once()
            self.assertIn("rubber", str(mock_logger.warning.call_args))

    def test_random_mode_with_rubber_no_warning(self):
        part_conf = {"sample": "random", "budget": 1000, "rubber": 42}

        with patch("training_data_packer.mode.lint.logger") as mock_logger:
            _check_sample_config(self.part_path, part_conf)
            mock_logger.warning.assert_not_called()

    def test_random_mode_with_filter_raises_error(self):
        part_conf = {"sample": "random", "budget": 1000, "filter": Path("tests/resources/sampler/test_sampler.py")}

        with self.assertRaises(ValueError) as context:
            _check_sample_config(self.part_path, part_conf)
        self.assertIn("random", str(context.exception))
        self.assertIn("filter", str(context.exception))

    def test_random_mode_with_parameters_raises_error(self):
        part_conf = {"sample": "random", "budget": 1000, "parameters": {"sample_ratio": "0.5"}}

        with self.assertRaises(ValueError) as context:
            _check_sample_config(self.part_path, part_conf)
        self.assertIn("random", str(context.exception))
        self.assertIn("parameters", str(context.exception))

    def test_random_mode_with_both_filter_and_parameters_raises_error(self):
        part_conf = {
            "sample": "random",
            "budget": 1000,
            "filter": Path("tests/resources/sampler/test_sampler.py"),
            "parameters": {"sample_ratio": "0.5"},
        }

        with self.assertRaises(ValueError) as context:
            _check_sample_config(self.part_path, part_conf)
        error_message = str(context.exception)
        self.assertIn("random", error_message)
        self.assertIn("filter", error_message)
        self.assertIn("parameters", error_message)

    def test_dynamic_mode_with_all_required_passes(self):
        part_conf = {
            "sample": "dynamic",
            "filter": Path("tests/resources/sampler/test_sampler.py"),
            "parameters": {"sample_ratio": "0.5"},
        }

        _check_sample_config(self.part_path, part_conf)
        self.assertTrue(True)

    def test_dynamic_mode_without_filter_raises_error(self):
        part_conf = {"sample": "dynamic", "parameters": {"sample_ratio": "0.5"}}

        with self.assertRaises(ValueError) as context:
            _check_sample_config(self.part_path, part_conf)
        self.assertIn("filter is missing", str(context.exception))
        self.assertIn(self.part_path, str(context.exception))

    def test_dynamic_mode_without_parameters_raises_error(self):
        part_conf = {"sample": "dynamic", "filter": Path("tests/resources/sampler/test_sampler.py")}

        with self.assertRaises(ValueError) as context:
            _check_sample_config(self.part_path, part_conf)
        self.assertIn("parameters is missing", str(context.exception))
        self.assertIn(self.part_path, str(context.exception))

    @patch("training_data_packer.mode.lint.read_sampler_fn")
    def test_dynamic_mode_with_invalid_filter_raises_error(self, mock_read_sampler):
        mock_read_sampler.side_effect = FileNotFoundError("Filter file not found")

        part_conf = {
            "sample": "dynamic",
            "filter": Path("non_existent_filter.py"),
            "parameters": {"sample_ratio": "0.5"},
        }

        with self.assertRaises(ValueError) as context:
            _check_sample_config(self.part_path, part_conf)
        self.assertIn("Fail to read sampler function", str(context.exception))
        self.assertIn(self.part_path, str(context.exception))

    def test_dynamic_mode_with_rubber_raises_error(self):
        part_conf = {
            "sample": "dynamic",
            "filter": Path("tests/resources/sampler/test_sampler.py"),
            "parameters": {"sample_ratio": "0.5"},
            "rubber": 42,
        }

        with self.assertRaises(ValueError) as context:
            _check_sample_config(self.part_path, part_conf)
        self.assertIn("dynamic", str(context.exception))
        self.assertIn("rubber", str(context.exception))

    def test_dynamic_mode_with_budget_raises_error(self):
        part_conf = {
            "sample": "dynamic",
            "filter": Path("tests/resources/sampler/test_sampler.py"),
            "parameters": {"sample_ratio": "0.5"},
            "budget": 1000,
        }

        with self.assertRaises(ValueError) as context:
            _check_sample_config(self.part_path, part_conf)
        self.assertIn("dynamic", str(context.exception))
        self.assertIn("budget", str(context.exception))

    def test_dynamic_mode_with_both_rubber_and_budget_raises_error(self):
        part_conf = {
            "sample": "dynamic",
            "filter": Path("tests/resources/sampler/test_sampler.py"),
            "parameters": {"sample_ratio": "0.5"},
            "rubber": 42,
            "budget": 1000,
        }

        with self.assertRaises(ValueError) as context:
            _check_sample_config(self.part_path, part_conf)
        error_message = str(context.exception)
        self.assertIn("dynamic", error_message)
        self.assertIn("rubber", error_message)
        self.assertIn("budget", error_message)

    def test_unknown_sample_mode_raises_error(self):
        part_conf = {"sample": "unknown_mode"}

        with self.assertRaises(ValueError) as context:
            _check_sample_config(self.part_path, part_conf)
        self.assertIn("unknown value", str(context.exception))
        self.assertIn("unknown_mode", str(context.exception))
        self.assertIn(self.part_path, str(context.exception))


class TestCheckPackConfig(unittest.TestCase):
    def setUp(self):
        self.part_path = "release.test-part"

    def test_flat_mode_without_prefix_warns(self):
        part_conf = {"pack": "flat"}

        with patch("training_data_packer.mode.lint.logger") as mock_logger:
            _check_pack_config(self.part_path, part_conf)
            mock_logger.warning.assert_called_once()
            self.assertIn("prefix", str(mock_logger.warning.call_args))

    def test_flat_mode_with_prefix_no_warning(self):
        part_conf = {"pack": "flat", "prefix": "custom_prefix"}

        with patch("training_data_packer.mode.lint.logger") as mock_logger:
            _check_pack_config(self.part_path, part_conf)
            mock_logger.warning.assert_not_called()

    def test_tree_mode_passes(self):
        part_conf = {"pack": "tree"}

        with patch("training_data_packer.mode.lint.logger") as mock_logger:
            _check_pack_config(self.part_path, part_conf)
            mock_logger.warning.assert_not_called()

    def test_unknown_pack_mode_raises_error(self):
        part_conf = {"pack": "unknown_mode"}

        with self.assertRaises(ValueError) as context:
            _check_pack_config(self.part_path, part_conf)
        self.assertIn("unknown value", str(context.exception))
        self.assertIn("unknown_mode", str(context.exception))
        self.assertIn(self.part_path, str(context.exception))

    def test_flat_mode_with_other_fields_passes(self):
        part_conf = {
            "pack": "flat",
            "prefix": "my_prefix",
            "suffix": ".jsonl",
            "sample": "full",
        }

        with patch("training_data_packer.mode.lint.logger") as mock_logger:
            _check_pack_config(self.part_path, part_conf)
            mock_logger.warning.assert_not_called()

    def test_tree_mode_with_other_fields_passes(self):
        part_conf = {
            "pack": "tree",
            "suffix": ".jsonl.gz",
            "sample": "random",
            "budget": 1000,
        }

        with patch("training_data_packer.mode.lint.logger") as mock_logger:
            _check_pack_config(self.part_path, part_conf)
            mock_logger.warning.assert_not_called()

    def test_flat_mode_with_empty_string_prefix_fail(self):
        part_conf = {"pack": "flat", "prefix": ""}

        with patch("training_data_packer.mode.lint.logger") as mock_logger:
            _check_pack_config(self.part_path, part_conf)
            mock_logger.warning.assert_called()

    def test_flat_mode_warning_includes_default_value(self):
        part_conf = {"pack": "flat"}

        with patch("training_data_packer.mode.lint.logger") as mock_logger:
            _check_pack_config(self.part_path, part_conf)
            warning_message = str(mock_logger.warning.call_args)
            self.assertIn("default", warning_message.lower())

    def test_tree_mode_with_prefix_passes(self):
        part_conf = {"pack": "tree", "prefix": "some_prefix"}

        with patch("training_data_packer.mode.lint.logger") as mock_logger:
            _check_pack_config(self.part_path, part_conf)
            mock_logger.warning.assert_not_called()

    def test_part_path_includes_in_warning_messages(self):
        part_conf = {"pack": "flat"}

        with patch("training_data_packer.mode.lint.logger") as mock_logger:
            _check_pack_config(self.part_path, part_conf)
            warning_message = str(mock_logger.warning.call_args)
            self.assertIn(self.part_path, warning_message)
