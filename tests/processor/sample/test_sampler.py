import unittest
from pathlib import Path

from parameterized import parameterized

from training_data_packer.processor.sample.sampler import DynamicSampler, convert_to_type


class TestDynamicSampler(unittest.TestCase):
    @parameterized.expand(
        [
            [
                "keep_record",
                {"id": "my_id", "text": "Some text", "sample": 1.0},
                [{"id": "my_id", "text": "Some text", "sample": 1.0}],
            ],
            [
                "remove_record",
                {"id": "my_id", "text": "Some text", "sample": 0.0},
                [],
            ],
            [
                "upsample_record",
                {"id": "my_id", "text": "Some text", "sample": 2.0},
                [
                    {"id": "my_id", "text": "Some text", "sample": 2.0},
                    {"id": "my_id", "text": "Some text", "sample": 2.0},
                ],
            ],
        ]
    )
    def test_dynamic_sampler(self, name, input_data, expected):
        dynamic_sampler = DynamicSampler(Path("tests/resources/sampler/test_sampler.py"), {"sample_ratio": "0"})
        mapper = dynamic_sampler.get_mapper()
        self.assertEqual(expected, mapper(input_data))
        self.assertEqual(0, dynamic_sampler.get_metrics()["DynamicSampler"]["sampler_ratio_exceptions"])

    def test_dynamic_sampler_with_exception(self):
        dynamic_sampler = DynamicSampler(Path("tests/resources/sampler/test_sampler_error.py"), {"sample_ratio": "0"})
        mapper = dynamic_sampler.get_mapper()
        in_data = {"id": "my_id", "text": "Some text", "sample": 1.0}
        self.assertEqual([in_data], mapper(in_data))
        self.assertEqual(1, dynamic_sampler.get_metrics()["DynamicSampler"]["sampler_ratio_exceptions"])


class TestConvertToType(unittest.TestCase):
    @parameterized.expand(
        [
            ["integers", {"a": "1", "b": "2", "c": "3"}, {"a": 1, "b": 2, "c": 3}],
            ["floats", {"a": "1.5", "b": "2.7", "c": "3.14"}, {"a": 1.5, "b": 2.7, "c": 3.14}],
            ["negative_integers", {"a": "-1", "b": "-2"}, {"a": -1, "b": -2}],
            ["negative_floats", {"a": "-1.5", "b": "-2.7"}, {"a": -1.5, "b": -2.7}],
            ["zero", {"a": "0", "b": "0.0"}, {"a": 0, "b": 0.0}],
            ["strings", {"a": "hello", "b": "world"}, {"a": "hello", "b": "world"}],
            [
                "mixed",
                {"a": "1", "b": "2.5", "c": "hello", "d": "3", "e": "4.2", "f": "world"},
                {"a": 1, "b": 2.5, "c": "hello", "d": 3, "e": 4.2, "f": "world"},
            ],
            ["empty_string", {"a": "", "b": "1"}, {"a": "", "b": 1}],
        ]
    )
    def test_convert_to_type(self, name, input_dict, expected):
        result = convert_to_type(input_dict)
        self.assertEqual(expected, result)

    def test_convert_to_type_preserves_dict(self):
        input_dict = {"a": "1", "b": "2"}
        result = convert_to_type(input_dict)
        self.assertEqual(input_dict, {"a": "1", "b": "2"})
        self.assertEqual(result, {"a": 1, "b": 2})
