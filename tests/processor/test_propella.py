import unittest

from training_data_packer.processor.propella import MergePropellaRecords, SourceToPropellaMapper


class TestPropellaProcessor(unittest.TestCase):
    def test_mapper(self):
        source_data = [
            {"my_id": "foo"},
            {"my_id": "bar"},
        ]
        id_field = "my_id"
        lookup_data = {
            "foo": {"my_id": "foo", "data": "foo_data"},
            "bar": None,  # Emulating not found.
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
        self.assertEqual(2, result_metrics["propella_matching"]["processed_records"])
        self.assertEqual(1, result_metrics["propella_matching"]["unmatched_records"])


class TestMergePropellaRecords(unittest.TestCase):
    def test_single_record_with_data(self):
        merger = MergePropellaRecords("id")
        mapper = merger.get_mapper()
        docs = [{"id": "123", "name": "John"}]
        result = mapper(docs)
        self.assertEqual({"id": "123", "name": "John"}, result)
        metrics = merger.get_metrics()
        self.assertEqual(1, metrics["propella_merge"]["processed_rows"])
        self.assertEqual(0, metrics["propella_merge"]["rows_with_duplicates"])
        self.assertEqual(0, metrics["propella_merge"]["rows_with_only_id"])

    def test_multiple_records_one_with_data(self):
        merger = MergePropellaRecords("id")
        mapper = merger.get_mapper()
        docs = [{"id": "123"}, {"id": "123", "name": "John"}, {"id": "123"}]
        result = mapper(docs)
        self.assertEqual({"id": "123", "name": "John"}, result)
        metrics = merger.get_metrics()
        self.assertEqual(1, metrics["propella_merge"]["processed_rows"])
        self.assertEqual(0, metrics["propella_merge"]["rows_with_duplicates"])
        self.assertEqual(0, metrics["propella_merge"]["rows_with_only_id"])

    def test_multiple_records_with_duplicate_data(self):
        merger = MergePropellaRecords("id")
        mapper = merger.get_mapper()
        docs = [{"id": "123", "name": "John"}, {"id": "123", "age": 30}]
        result = mapper(docs)
        self.assertEqual({"id": "123"}, result)
        metrics = merger.get_metrics()
        self.assertEqual(1, metrics["propella_merge"]["processed_rows"])
        self.assertEqual(1, metrics["propella_merge"]["rows_with_duplicates"])
        self.assertEqual(0, metrics["propella_merge"]["rows_with_only_id"])

    def test_all_records_only_id(self):
        merger = MergePropellaRecords("id")
        mapper = merger.get_mapper()
        docs = [{"id": "123"}, {"id": "123"}, {"id": "123"}]
        result = mapper(docs)
        self.assertEqual({"id": "123"}, result)
        metrics = merger.get_metrics()
        self.assertEqual(1, metrics["propella_merge"]["processed_rows"])
        self.assertEqual(0, metrics["propella_merge"]["rows_with_duplicates"])
        self.assertEqual(1, metrics["propella_merge"]["rows_with_only_id"])

    def test_misaligned_ids_raises_error(self):
        merger = MergePropellaRecords("id")
        mapper = merger.get_mapper()
        docs = [{"id": "123"}, {"id": "456"}]
        with self.assertRaises(ValueError) as context:
            mapper(docs)
        self.assertEqual("Files not aligned", str(context.exception))

    def test_custom_id_field(self):
        merger = MergePropellaRecords("custom_id")
        mapper = merger.get_mapper()
        docs = [{"custom_id": "123", "name": "John"}]
        result = mapper(docs)
        self.assertEqual({"custom_id": "123", "name": "John"}, result)

    def test_custom_metric_name(self):
        merger = MergePropellaRecords("id", metric_name="custom_metric")
        mapper = merger.get_mapper()
        docs = [{"id": "123", "name": "John"}]
        result = mapper(docs)
        self.assertEqual({"id": "123", "name": "John"}, result)
        metrics = merger.get_metrics()
        self.assertIn("custom_metric", metrics)

    def test_multiple_calls_accumulate_metrics(self):
        merger = MergePropellaRecords("id")
        mapper = merger.get_mapper()

        mapper([{"id": "123", "name": "John"}])
        mapper([{"id": "456"}])
        mapper([{"id": "789", "age": 30}, {"id": "789", "city": "NYC"}])

        metrics = merger.get_metrics()
        self.assertEqual(3, metrics["propella_merge"]["processed_rows"])
        self.assertEqual(1, metrics["propella_merge"]["rows_with_duplicates"])
        self.assertEqual(1, metrics["propella_merge"]["rows_with_only_id"])


if __name__ == "__main__":
    unittest.main()
