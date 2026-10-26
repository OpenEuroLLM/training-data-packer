"""Tokenize JSONL files outside of a collection directory."""

import argparse
from pathlib import Path

from loguru import logger

from training_data_packer.metadata import Metadata
from training_data_packer.metadata.defaults import DEFAULT_SUFFIX, DEFAULT_TEXT
from training_data_packer.mode.tokenize import tokenize_file
from training_data_packer.utils.file import find_files

_KNOWN_SUFFIXES = [".jsonl.zst", ".jsonl.zstd", ".jsonl.gz", ".jsonl", ".parquet"]


def process(input_path: Path, tokenizer: str, text_field: str = DEFAULT_TEXT, output_dir: Path = Path(".")) -> None:
    """Tokenize a file, or every file in a directory, into indexed datasets.

    For a directory, every ``*.jsonl.zst`` file below it is tokenized and the output keeps the relative path of the
    input file under ``output_dir``. For a single file, the output is written directly to ``output_dir``.

    :param input_path: the file or directory to tokenize
    :param tokenizer: HuggingFace Hub repo id or local path of the tokenizer
    :param text_field: the record field holding the text to tokenize
    :param output_dir: the directory to write the tokenized datasets to
    :raises ValueError: if no input files are found
    """
    if input_path.is_dir():
        src_files = find_files(input_path, DEFAULT_SUFFIX)
        base_dir = input_path
    else:
        src_files = [input_path] if input_path.exists() else []
        base_dir = input_path.parent
    if len(src_files) == 0:
        raise ValueError(f"No files found in {input_path}")
    logger.info(f"Found {len(src_files)} files")

    metadata = Metadata({"tokenizer": tokenizer})
    for src_file in src_files:
        output_file_prefix = _output_file_prefix(src_file, base_dir, output_dir)
        logger.info(f"Tokenizing {src_file} to {output_file_prefix}")
        tokenize_file(src_file, metadata, text_field=text_field, output_file_prefix=output_file_prefix)


def _output_file_prefix(src_file: Path, base_dir: Path, output_dir: Path) -> Path:
    rel_path = src_file.relative_to(base_dir)
    name = rel_path.name
    for suffix in _KNOWN_SUFFIXES:
        if name.endswith(suffix):
            name = name.removesuffix(suffix)
            break
    return output_dir / rel_path.with_name(name)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="oellm-tokenize",
        description="Tokenize a JSONL file, or all *.jsonl.zst files in a directory, into Megatron indexed datasets.",
    )
    parser.add_argument("input", help="File or directory to tokenize")
    parser.add_argument("-t", "--tokenizer", help="HuggingFace Hub repo id or local tokenizer path", required=True)
    parser.add_argument("--text", help=f"Field holding the text, default is {DEFAULT_TEXT}", default=DEFAULT_TEXT)
    parser.add_argument("-o", "--output-dir", help="Output directory, default is current directory", default=".")
    args = parser.parse_args()

    process(Path(args.input), args.tokenizer, args.text, Path(args.output_dir))


if __name__ == "__main__":
    main()
