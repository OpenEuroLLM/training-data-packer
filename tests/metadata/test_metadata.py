import unittest
from pathlib import Path

from parameterized import parameterized

from training_data_packer.metadata import (
    Metadata,
    calculate_file_path,
    get_matching_part,
    get_megatron_lm_tokenizers,
    get_shard_size_documents,
)


class TestMetadata(unittest.TestCase):
    def test_metadata_without_defaults(self):
        indata = Metadata({"release": {"foo": {"sample": "full"}, "bar": {"sample": "full"}}})
        self.assertEqual(get_matching_part(indata, Path("bla/foo/shard01")), ({"sample": "full"}, "foo"))

    def test_metadata_merge_defaults(self):
        indata = Metadata(
            {"release": {"default": {"pack": "tree"}, "foo": {"sample": "full"}, "bar": {"sample": "full"}}}
        )
        self.assertEqual(
            get_matching_part(indata, Path("bla/foo/shard01")), ({"sample": "full", "pack": "tree"}, "foo")
        )

    def test_metadata_use_defaults(self):
        indata = Metadata(
            {
                "release": {
                    "default": {
                        "pack": "tree",
                        "sample": "wds",
                    },
                    "foo": {},
                    "bar": {
                        "sample": "full",
                    },
                }
            }
        )
        self.assertEqual(get_matching_part(indata, Path("bla/foo/shard01")), ({"sample": "wds", "pack": "tree"}, "foo"))

    def test_metadata_use_defaults_look_in_hierarchy(self):
        indata = Metadata(
            {
                "source": {
                    "foo": {"value": "do not use"},
                },
                "release": {
                    "default": {
                        "input": "source",
                        "pack": "tree",
                        "sample": "wds",
                    },
                    "bar": {
                        "sample": "full",
                    },
                },
            }
        )
        self.assertEqual(
            get_matching_part(indata, Path("bla/foo/shard01")),
            ({"input": "source", "sample": "wds", "pack": "tree"}, "foo"),
        )

    def test_metadata_fail_find_in_hierarchy(self):
        indata = Metadata(
            {
                "source": {},
                "release": {
                    "default": {
                        "input": "source",
                        "pack": "tree",
                        "sample": "wds",
                    },
                    "bar": {
                        "sample": "full",
                    },
                },
            }
        )
        self.assertEqual((None, None), get_matching_part(indata, Path("bla/foo/shard01")))

    def test_metadata_override_defaults(self):
        indata = Metadata(
            {
                "release": {
                    "default": {
                        "pack": "tree",
                        "sample": "wds",
                    },
                    "foo": {
                        "sample": "full",
                    },
                    "bar": {
                        "sample": "full",
                    },
                }
            }
        )
        self.assertEqual(
            get_matching_part(indata, Path("bla/foo/shard01")), ({"sample": "full", "pack": "tree"}, "foo")
        )

    def test_no_matching_part(self):
        indata = Metadata(
            {
                "source": {
                    "default": {
                        "pack": "tree",
                        "sample": "wds",
                    },
                    "bar": {
                        "sample": "full",
                    },
                }
            }
        )
        self.assertEqual((None, None), get_matching_part(indata, Path("bla/foo/shard01"), "source"))

    @parameterized.expand(
        [
            [
                "flat",
                Metadata(
                    {
                        "_internal": {"mode": "release"},
                        "release": {
                            "default": {
                                "unimportant": 5,
                            },
                            "part2": {},
                            "part1": {},
                        },
                    }
                ),
                ["part1", "part2"],
            ],
            [
                "hierarchy",
                Metadata(
                    {
                        "_internal": {"mode": "release"},
                        "release": {
                            "default": {"unimportant": 5, "input": "another"},
                            "part2": {},
                            "part1": {},
                        },
                        "another": {
                            "part3": {},
                            "part1": {},
                        },
                    }
                ),
                ["part1", "part2", "part3"],
            ],
        ]
    )
    def test_get_all_part_names(self, name: str, metadata: Metadata, expected: list):
        self.assertEqual(expected, metadata.get_all_part_names("release"))

    def test_get_shard_size_documents(self):
        self.assertEqual(10_000_000_000, get_shard_size_documents({"shard": "10bd"}))
        self.assertEqual(5_000_000, get_shard_size_documents({"shard": "5md"}))
        self.assertEqual(350, get_shard_size_documents({"shard": "350d"}))
        with self.assertRaises(ValueError):
            get_shard_size_documents({"shard": "350tt"})
        with self.assertRaises(ValueError):
            get_shard_size_documents({"shard": "350"})

    @parameterized.expand(
        [
            ["Value_set", Metadata({"foo": "bar"}), "foo", "gazsonk", "bar"],
            ["default", {"foo": "bar"}, "key", "gazonk", "gazonk"],
            [
                "hierarchy",
                Metadata({"foo": {"bar": 42}}),
                "foo.bar",
                17,
                42,
            ],
            [
                "array_square_bracket",
                Metadata({"a": [{"b": 5}, {"b": 7}]}),
                "a[0].b",
                17,
                5,
            ],
            [
                "field_with_dot",
                Metadata({"a": [{"b": 5}, {"b.c": 7}]}),
                "a[1].'b.c'",
                17,
                7,
            ],
        ]
    )
    def test_get_metadata_value(self, name, metadata, key, default_value, expected):
        result = metadata.get(key, default_value)
        self.assertEqual(expected, result)


if __name__ == "__main__":
    unittest.main()


class TestGetMegatronLmTokenizers(unittest.TestCase):
    def test_merges_default(self):
        metadata = Metadata(
            {
                "megatron-lm": {
                    "default": {"source": "release"},
                    "first": {"huggingface": "org/first"},
                    "second": {"huggingface": "org/second", "source": "release-raw"},
                },
            },
        )
        self.assertEqual(
            {
                "first": {"huggingface": "org/first", "source": "release", "bos": "<bos>", "eos": "<eos>"},
                "second": {"huggingface": "org/second", "source": "release-raw", "bos": "<bos>", "eos": "<eos>"},
            },
            get_megatron_lm_tokenizers(metadata),
        )

    def test_bos_and_eos_set_in_default_and_tokenizer(self):
        metadata = {
            "megatron-lm": {
                "default": {"source": "release", "bos": False},
                "first": {"huggingface": "org/first"},
                "second": {"huggingface": "org/second", "bos": "<s>", "eos": False},
            },
        }
        tokenizers = get_megatron_lm_tokenizers(metadata)
        self.assertEqual((False, "<eos>"), (tokenizers["first"]["bos"], tokenizers["first"]["eos"]))
        self.assertEqual(("<s>", False), (tokenizers["second"]["bos"], tokenizers["second"]["eos"]))

    def test_keeps_metadata_order(self):
        metadata = {
            "megatron-lm": {"default": {"source": "release"}, "b": {"huggingface": "b"}, "a": {"huggingface": "a"}}
        }
        self.assertEqual(["b", "a"], list(get_megatron_lm_tokenizers(metadata)))

    @parameterized.expand(
        [
            ["missing section", {}],
            ["only default", {"megatron-lm": {"default": {"source": "release"}}}],
        ]
    )
    def test_no_tokenizers(self, _name: str, metadata: dict):
        self.assertEqual({}, get_megatron_lm_tokenizers(metadata))


class TestCalculateFilePath(unittest.TestCase):
    src_dir = Path("/srv/data/gazonk/source")
    contamination_dir = Path("/srv/data/gazonk/contamination")
    pii_dir = Path("/srv/data/gazonk/pii")
    propella_dir = Path("/srv/data/gazonk/propella-4b")
    output_dir = Path("/srv/data/gazonk/release_raw")

    def test_calculate_file_paths_default(self):
        src_file = self.src_dir.joinpath("shard01/file01.jsonl.zst")
        metadata = Metadata(
            {
                "_internal": {"collection_dir": Path("/srv/data/gazonk"), "mode": "release"},
                "release": {"default": {"input": "source"}},
                "suffix": ".jsonl.zstd",
            }
        )
        contamination_file = calculate_file_path(src_file, metadata, self.contamination_dir)
        pii_file = calculate_file_path(src_file, metadata, self.pii_dir)
        propella_file = calculate_file_path(src_file, metadata, self.propella_dir)
        out_file = calculate_file_path(src_file, metadata, self.output_dir)

        self.assertEqual(self.contamination_dir.joinpath("shard01/file01.jsonl.zst"), contamination_file)
        self.assertEqual(self.pii_dir.joinpath("shard01/file01.jsonl.zst"), pii_file)
        self.assertEqual(self.propella_dir.joinpath("shard01/file01.jsonl.zst"), propella_file)
        self.assertEqual(self.output_dir.joinpath("shard01/file01.jsonl.zst"), out_file)

    def test_calculate_file_paths_source_suffix(self):
        src_file = self.src_dir.joinpath("shard01/file01.jsonl.gz")
        metadata = Metadata(
            {
                "_internal": {"collection_dir": Path("/srv/data/gazonk"), "mode": "release"},
                "release": {"default": {"input": "source"}},
                "source": {"default": {"suffix": ".jsonl.gz"}},
                "suffix": ".jsonl.zst",
            }
        )

        contamination_file = calculate_file_path(src_file, metadata, self.contamination_dir)
        pii_file = calculate_file_path(src_file, metadata, self.pii_dir)
        propella_file = calculate_file_path(src_file, metadata, self.propella_dir)
        out_file = calculate_file_path(src_file, metadata, self.output_dir)

        self.assertEqual(self.contamination_dir.joinpath("shard01/file01.jsonl.zst"), contamination_file)
        self.assertEqual(self.pii_dir.joinpath("shard01/file01.jsonl.zst"), pii_file)
        self.assertEqual(self.propella_dir.joinpath("shard01/file01.jsonl.zst"), propella_file)
        self.assertEqual(self.output_dir.joinpath("shard01/file01.jsonl.zst"), out_file)

    def test_calculate_file_paths_sample_source_suffix(self):
        src_file = self.src_dir.joinpath("shard01/file01.jsonl.gz")
        output_dir = Path("/srv/data/gazonk/sample")
        metadata = Metadata(
            {
                "_internal": {"collection_dir": Path("/srv/data/gazonk"), "mode": "sample"},
                "sample": {"default": {"input": "source"}},
                "source": {"default": {"suffix": ".jsonl.gz"}},
                "suffix": ".jsonl.zst",
            }
        )

        contamination_file = calculate_file_path(src_file, metadata, self.contamination_dir)
        pii_file = calculate_file_path(src_file, metadata, self.pii_dir)
        propella_file = calculate_file_path(src_file, metadata, self.propella_dir)
        out_file = calculate_file_path(src_file, metadata, output_dir)

        self.assertEqual(self.contamination_dir.joinpath("shard01/file01.jsonl.zst"), contamination_file)
        self.assertEqual(self.pii_dir.joinpath("shard01/file01.jsonl.zst"), pii_file)
        self.assertEqual(self.propella_dir.joinpath("shard01/file01.jsonl.zst"), propella_file)
        self.assertEqual(output_dir.joinpath("shard01/file01.jsonl.zst"), out_file)
