import unittest

from training_data_packer.processor.propella import SourceToPropellaMapper


class TestPropellaProcessor(unittest.TestCase):
    def test_mapper(self):
        source_data = [
            {"my_id": "foo"},
            {"my_id": "bar"},
        ]
        id_field = "my_id"
        lookup_data = {
            "foo": [{"my_id": "foo", "data": "foo_data"}],
            "bar": [],  # Emulating not found.
        }
        source_to_propella_mapper = SourceToPropellaMapper(id_field, lambda x: lookup_data[x])
        result = list(map(source_to_propella_mapper.get_mapper(), source_data))
        self.assertEqual(
            [
                {"my_id": "foo", "data": "foo_data"},
                {"my_id": "bar"},
            ],
            result,
        )
        result_metrics = source_to_propella_mapper.get_metrics()
        self.assertEqual(2, result_metrics["doc_ordering"]["processed_records"])
        self.assertEqual(1, result_metrics["doc_ordering"]["unmatched_records"])


if __name__ == "__main__":
    unittest.main()
