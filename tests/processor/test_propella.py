import unittest

from parameterized import parameterized

from training_data_packer.processor.propella import (
    MergePropellaRecords,
    SourceToPropellaMapper,
    propella_annotate_factory,
)


class TestPropellaProcessor(unittest.TestCase):
    def test_mapper(self):
        source_data = [
            {"my_id": "foo"},
            {"my_id": "bar"},
        ]
        metadata = {"id": "my_id", "text": "text"}
        lookup_data = {
            "foo": {"id": "foo", "data": "foo_data"},
            "bar": None,  # Emulating not found.
        }
        source_to_propella_mapper = SourceToPropellaMapper(metadata, lambda x: lookup_data[x])
        result = list(map(source_to_propella_mapper.get_mapper(), source_data))
        self.assertEqual(
            [
                {"id": "foo", "propella-4b": {"id": "foo", "data": "foo_data"}},
                {"id": "bar"},
            ],
            result,
        )
        result_metrics = source_to_propella_mapper.get_metrics()
        self.assertEqual(2, result_metrics["propella_matching"]["processed_records"])
        self.assertEqual(1, result_metrics["propella_matching"]["unmatched_records"])

    def test_mapper_with_id_hash(self):
        source_data = [
            {"id": "foo", "text": "this is my document"},
            {"id": "bar", "text": "this is my second document"},
        ]
        metadata = {
            "id": "id",
            "text": "text",
            "annotations": {
                "propella-4b": {"hash": "sha256"},
            },
        }
        lookup_data = {
            "0422bdf3f65ea9ebda3004d0b4e392906c10f5591a32034eeb41f86b78919d4d": {
                "id": "0422bdf3f65ea9ebda3004d0b4e392906c10f5591a32034eeb41f86b78919d4d",
                "data": "foo_data",
            },
            "6c0bf0ef0125de8874197d25e9e5cb04fd07f70917060b40025eb3ef8b7369cf": None,  # Emulating not found.
        }
        source_to_propella_mapper = SourceToPropellaMapper(metadata, lambda x: lookup_data[x])
        result = list(map(source_to_propella_mapper.get_mapper(), source_data))
        self.assertEqual(
            [
                {
                    "id": "foo",
                    "hash": "0422bdf3f65ea9ebda3004d0b4e392906c10f5591a32034eeb41f86b78919d4d",
                    "propella-4b": {
                        "id": "0422bdf3f65ea9ebda3004d0b4e392906c10f5591a32034eeb41f86b78919d4d",
                        "data": "foo_data",
                    },
                },
                {"id": "bar", "hash": "6c0bf0ef0125de8874197d25e9e5cb04fd07f70917060b40025eb3ef8b7369cf"},
            ],
            result,
        )
        result_metrics = source_to_propella_mapper.get_metrics()
        self.assertEqual(2, result_metrics["propella_matching"]["processed_records"])
        self.assertEqual(1, result_metrics["propella_matching"]["unmatched_records"])

    def test_mapper_with_id_hash_and_length(self):
        source_data = [
            {"id": "foo", "text": "this is my document"},
            {"id": "bar", "text": "this is my second document"},
        ]
        metadata = {
            "id": "id",
            "text": "text",
            "annotations": {
                "propella-4b": {"hash": "sha256-32", "hash-id": "hash_id"},
            },
        }
        lookup_data = {
            "0422bdf3f65ea9ebda3004d0b4e39290": {
                "id": "0422bdf3f65ea9ebda3004d0b4e39290",
                "data": "foo_data",
            },
            "6c0bf0ef0125de8874197d25e9e5cb04": None,  # Emulating not found.
        }
        source_to_propella_mapper = SourceToPropellaMapper(metadata, lambda x: lookup_data[x])
        result = list(map(source_to_propella_mapper.get_mapper(), source_data))
        self.assertEqual(
            [
                {
                    "id": "foo",
                    "hash_id": "0422bdf3f65ea9ebda3004d0b4e39290",
                    "propella-4b": {
                        "id": "0422bdf3f65ea9ebda3004d0b4e39290",
                        "data": "foo_data",
                    },
                },
                {"id": "bar", "hash_id": "6c0bf0ef0125de8874197d25e9e5cb04"},
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
        docs = [{"id": "123", "propella-4b": {"name": "John"}}]
        result = mapper(docs)
        self.assertEqual({"id": "123", "propella-4b": {"name": "John"}}, result)
        metrics = merger.get_metrics()
        self.assertEqual(1, metrics["propella_merge"]["processed_rows"])
        self.assertEqual(0, metrics["propella_merge"]["rows_with_duplicates"])
        self.assertEqual(0, metrics["propella_merge"]["rows_with_only_id"])

    def test_multiple_records_one_with_data(self):
        merger = MergePropellaRecords("id")
        mapper = merger.get_mapper()
        docs = [{"id": "123"}, {"id": "123", "propella-4b": {"name": "John"}}, {"id": "123"}]
        result = mapper(docs)
        self.assertEqual({"id": "123", "propella-4b": {"name": "John"}}, result)
        metrics = merger.get_metrics()
        self.assertEqual(1, metrics["propella_merge"]["processed_rows"])
        self.assertEqual(0, metrics["propella_merge"]["rows_with_duplicates"])
        self.assertEqual(0, metrics["propella_merge"]["rows_with_only_id"])

    def test_multiple_records_with_duplicate_data(self):
        merger = MergePropellaRecords("id")
        mapper = merger.get_mapper()
        docs = [{"id": "123", "propella-4b": {"name": "John"}}, {"id": "123", "propella-4b": {"age": 30}}]
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

    def test_all_records_only_id_and_hash(self):
        merger = MergePropellaRecords("id")
        mapper = merger.get_mapper()
        docs = [{"id": "123", "hash": "#1"}, {"id": "123", "hash": "#1"}, {"id": "123", "hash": "#1"}]
        result = mapper(docs)
        self.assertEqual({"id": "123", "hash": "#1"}, result)
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
        docs = [{"custom_id": "123", "propella-4b": {"name": "John"}}]
        result = mapper(docs)
        self.assertEqual({"custom_id": "123", "propella-4b": {"name": "John"}}, result)

    def test_hash_field(self):
        merger = MergePropellaRecords("id")
        mapper = merger.get_mapper()
        docs = [{"id": "123", "hash": "###", "propella-4b": {"name": "John"}}]
        result = mapper(docs)
        self.assertEqual({"id": "123", "hash": "###", "propella-4b": {"name": "John"}}, result)

    def test_custom_metric_name(self):
        merger = MergePropellaRecords("id", metric_name="custom_metric")
        mapper = merger.get_mapper()
        docs = [{"id": "123", "propella-4b": {"name": "John"}}]
        result = mapper(docs)
        self.assertEqual({"id": "123", "propella-4b": {"name": "John"}}, result)
        metrics = merger.get_metrics()
        self.assertIn("custom_metric", metrics)

    def test_multiple_calls_accumulate_metrics(self):
        merger = MergePropellaRecords("id")
        mapper = merger.get_mapper()

        mapper([{"id": "123", "propella-4b": {"name": "John"}}])
        mapper([{"id": "456"}])
        mapper([{"id": "789", "propella-4b": {"age": 30}}, {"id": "789", "propella-4b": {"city": "NYC"}}])

        metrics = merger.get_metrics()
        self.assertEqual(3, metrics["propella_merge"]["processed_rows"])
        self.assertEqual(1, metrics["propella_merge"]["rows_with_duplicates"])
        self.assertEqual(1, metrics["propella_merge"]["rows_with_only_id"])


class TestPropellaAnnotateFactory(unittest.TestCase):
    @parameterized.expand(
        [
            [
                "merge",
                [{"id": "1234", "text": "Happy"}],
                [{"id": "1234", "sample": "full"}],
                [{"id": "1234", "text": "Happy", "propella-4b": {"sample": "full"}}],
            ],
            [
                "only-id",
                [{"id": "1234", "text": "Happy"}],
                [{"id": "1234"}],
                [{"id": "1234", "text": "Happy"}],
            ],
            [
                "None",
                [{"id": "1234", "text": "Happy"}],
                None,
                [{"id": "1234", "text": "Happy"}],
            ],
        ]
    )
    def test_no_mapping_needed(self, name, in_iter, prop_iter, expected):
        result_iter = propella_annotate_factory(in_iter, prop_iter)
        result_list = list(result_iter)
        self.assertEqual(expected, result_list)


if __name__ == "__main__":
    unittest.main()
