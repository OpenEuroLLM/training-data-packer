import unittest

from parameterized import parameterized

from training_data_packer.processor.sample.sampler import DynamicSampler


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
        dynamic_sampler = DynamicSampler("tests/resources/sampler/test_sampler.py")
        mapper = dynamic_sampler.get_mapper()
        self.assertEqual(expected, mapper(input_data))
