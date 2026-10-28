import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from tests.helpers.tokenizer import write_tiny_tokenizer
from training_data_packer.mode import tokenize
from training_data_packer.tokenizer.indexed_dataset import read_documents
from training_data_packer.utils.file import JsonlZstWriter

# Token ids of the tiny tokenizer: bytes map to themselves, "ab" is 256, <eos> is 258 and <bos> is 259.
_EOS_ID = 258
_BOS_ID = 259


def _read_tokens(prefix: Path) -> list[list[int]]:
    return [doc.tolist() for doc in read_documents(prefix)]


class TestOutputFilePrefix(unittest.TestCase):
    def test_keeps_relative_path_and_removes_suffix(self):
        result = tokenize._output_file_prefix(
            Path("/c/release/lang/shard01/000.jsonl.zst"),
            Path("/c/release"),
            Path("/c/megatron-lm/tok"),
        )
        self.assertEqual(Path("/c/megatron-lm/tok/lang/shard01/000"), result)


class TestProcess(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.collection_dir = Path(self._tmp.name)
        self.tokenizer = str(write_tiny_tokenizer(self.collection_dir / "tokenizer.json", with_bos=True))
        patcher = mock.patch("training_data_packer.utils.slurm.is_slurm_job", return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self._tmp.cleanup()

    def _write_metadata(self, megatron_lm: dict) -> None:
        metadata = {
            "name": "test",
            "release": {"default": {"input": "source", "pack": "flat"}},
            "megatron-lm": megatron_lm,
        }
        (self.collection_dir / "metadata.yaml").write_text(yaml.safe_dump(metadata))

    def _write_input(self, rel_path: str, records: list[dict]) -> None:
        path = self.collection_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        JsonlZstWriter(path).write(iter(records))

    def test_single_tokenizer(self):
        self._write_metadata({"default": {"source": "release"}, "tiny": {"huggingface": self.tokenizer}})
        self._write_input("release/shard01/000.jsonl.zst", [{"text": "ab"}, {"text": "c"}])

        tokenize.process(self.collection_dir)

        self.assertEqual(
            [[_BOS_ID, 256, _EOS_ID], [_BOS_ID, ord("c"), _EOS_ID]],
            _read_tokens(self.collection_dir / "megatron-lm" / "tiny" / "shard01" / "000"),
        )

    def test_multiple_tokenizers(self):
        self._write_metadata(
            {
                "default": {"source": "release"},
                "first": {"huggingface": self.tokenizer},
                "second": {"huggingface": self.tokenizer},
            },
        )
        self._write_input("release/shard01/000.jsonl.zst", [{"text": "ab"}])

        tokenize.process(self.collection_dir)

        for name in ["first", "second"]:
            self.assertEqual(
                [[_BOS_ID, 256, _EOS_ID]],
                _read_tokens(self.collection_dir / "megatron-lm" / name / "shard01" / "000"),
            )

    def test_tokenizer_overrides_source(self):
        self._write_metadata(
            {"default": {"source": "release"}, "tiny": {"huggingface": self.tokenizer, "source": "release-raw"}},
        )
        self._write_input("release-raw/shard01/000.jsonl.zst", [{"text": "c"}])

        tokenize.process(self.collection_dir)

        self.assertEqual(
            [[_BOS_ID, ord("c"), _EOS_ID]],
            _read_tokens(self.collection_dir / "megatron-lm" / "tiny" / "shard01" / "000"),
        )

    def test_no_files_raises(self):
        self._write_metadata({"default": {"source": "release"}, "tiny": {"huggingface": self.tokenizer}})

        with self.assertRaises(ValueError):
            tokenize.process(self.collection_dir)


if __name__ == "__main__":
    unittest.main()
