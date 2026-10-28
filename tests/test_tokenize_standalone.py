import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from parameterized import parameterized

from tests.helpers.tokenizer import write_tiny_tokenizer
from training_data_packer import tokenize_standalone
from training_data_packer.tokenizer.indexed_dataset import metrics_path, read_documents
from training_data_packer.utils.file import JsonlZstWriter

# Token ids of the tiny tokenizer: bytes map to themselves, "ab" is 256, <eos> is 258 and <bos> is 259.
_EOS_ID = 258
_BOS_ID = 259


def _read_tokens(prefix: Path) -> list[list[int]]:
    return [doc.tolist() for doc in read_documents(prefix)]


class TestOutputFilePrefix(unittest.TestCase):
    @parameterized.expand(
        [
            ("zst", "/in/000.jsonl.zst", "/in", "/out", "/out/000"),
            ("zstd", "/in/000.jsonl.zstd", "/in", "/out", "/out/000"),
            ("gz", "/in/000.jsonl.gz", "/in", "/out", "/out/000"),
            ("plain", "/in/000.jsonl", "/in", "/out", "/out/000"),
            ("parquet", "/in/000.parquet", "/in", "/out", "/out/000"),
            ("nested", "/in/deu_Latn/shard/000.jsonl.zst", "/in", "/out", "/out/deu_Latn/shard/000"),
            ("dots_in_name", "/in/a.b.jsonl.zst", "/in", "/out", "/out/a.b"),
            ("unknown_suffix", "/in/000.txt", "/in", "/out", "/out/000.txt"),
        ],
    )
    def test_output_file_prefix(self, _name, src_file, base_dir, output_dir, expected):
        result = tokenize_standalone._output_file_prefix(Path(src_file), Path(base_dir), Path(output_dir))
        self.assertEqual(Path(expected), result)


class TestProcess(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        self.tokenizer = str(write_tiny_tokenizer(self.tmp_dir / "tokenizer.json", with_bos=True))
        self.input_dir = self.tmp_dir / "input"
        self.output_dir = self.tmp_dir / "output"

    def tearDown(self):
        self._tmp.cleanup()

    def _write_input(self, rel_path: str, records: list[dict]) -> Path:
        path = self.input_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        JsonlZstWriter(path).write(iter(records))
        return path

    def test_directory(self):
        self._write_input("000.jsonl.zst", [{"text": "ab"}, {"text": "abc"}])
        self._write_input("part/001.jsonl.zst", [{"text": "c"}])

        tokenize_standalone.process(self.input_dir, self.tokenizer, output_dir=self.output_dir)

        self.assertEqual(
            [[_BOS_ID, 256, _EOS_ID], [_BOS_ID, 256, ord("c"), _EOS_ID]], _read_tokens(self.output_dir / "000")
        )
        self.assertEqual([[_BOS_ID, ord("c"), _EOS_ID]], _read_tokens(self.output_dir / "part" / "001"))

    def test_single_file(self):
        src_file = self._write_input("deep/000.jsonl.zst", [{"text": "ab"}])

        tokenize_standalone.process(src_file, self.tokenizer, output_dir=self.output_dir)

        self.assertEqual([[_BOS_ID, 256, _EOS_ID]], _read_tokens(self.output_dir / "000"))

    def test_text_field(self):
        self._write_input("000.jsonl.zst", [{"text": "ignored", "content": "ab"}])

        tokenize_standalone.process(self.input_dir, self.tokenizer, text_field="content", output_dir=self.output_dir)

        self.assertEqual([[_BOS_ID, 256, _EOS_ID]], _read_tokens(self.output_dir / "000"))
        metrics = json.loads(metrics_path(self.output_dir / "000").read_text())
        self.assertEqual("content", metrics["text"])
        self.assertEqual(1, metrics["documents"])
        self.assertEqual(3, metrics["tokens"])

    @parameterized.expand(
        [
            ("task_0", "0", ["000", "002"], ["001"]),
            ("task_1", "1", ["001"], ["000", "002"]),
        ],
    )
    def test_slurm_processes_only_own_files(self, _name, task_id, expected, not_expected):
        for name in ["000", "001", "002"]:
            self._write_input(f"{name}.jsonl.zst", [{"text": "ab"}])

        env = {"SLURM_ARRAY_TASK_ID": task_id, "SLURM_ARRAY_TASK_COUNT": "2"}
        with mock.patch.dict("os.environ", env):
            tokenize_standalone.process(self.input_dir, self.tokenizer, output_dir=self.output_dir, slurm_job=True)

        for name in expected:
            self.assertEqual([[_BOS_ID, 256, _EOS_ID]], _read_tokens(self.output_dir / name))
        for name in not_expected:
            self.assertFalse(self.output_dir.joinpath(f"{name}.bin").exists())

    def test_missing_text_field_raises(self):
        self._write_input("000.jsonl.zst", [{"content": "ab"}])

        with self.assertRaises(KeyError):
            tokenize_standalone.process(self.input_dir, self.tokenizer, output_dir=self.output_dir)

    @parameterized.expand(
        [
            ("empty_directory", "input"),
            ("missing_file", "input/missing.jsonl.zst"),
        ],
    )
    def test_no_files_raises(self, _name, rel_path):
        self.input_dir.mkdir()

        with self.assertRaises(ValueError):
            tokenize_standalone.process(self.tmp_dir / rel_path, self.tokenizer, output_dir=self.output_dir)

    def test_main_parses_arguments(self):
        argv = [
            "oellm-tokenize-standalone",
            str(self.input_dir),
            "-t",
            self.tokenizer,
            "--text",
            "content",
            "-o",
            "out",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(tokenize_standalone, "process") as process:
            tokenize_standalone.main()

        process.assert_called_once_with(self.input_dir, self.tokenizer, "content", Path("out"), False)

    def test_main_defaults(self):
        argv = ["oellm-tokenize-standalone", str(self.input_dir), "-t", self.tokenizer]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(tokenize_standalone, "process") as process:
            tokenize_standalone.main()

        process.assert_called_once_with(self.input_dir, self.tokenizer, "text", Path("."), False)

    def test_main_slurm(self):
        argv = ["oellm-tokenize-standalone", str(self.input_dir), "-t", self.tokenizer, "--slurm"]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(tokenize_standalone, "process") as process:
            tokenize_standalone.main()

        process.assert_called_once_with(self.input_dir, self.tokenizer, "text", Path("."), True)


if __name__ == "__main__":
    unittest.main()
