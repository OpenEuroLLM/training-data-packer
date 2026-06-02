import gzip
import io
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any

import orjson as json
import zstandard as zstd
from loguru import logger


def find_files(source_dir: Path, metadata: dict, part: str = None) -> list[Path]:
    """
    Returns all files, not hidden, under source_dir, following symlinks
    :param source_dir: Dir to find files in
    :param metadata: Metadata information to get expected file suffix.
    :param part: Optional part if only parts of source_dir are needed.
    :return: List of Path objects to files. Sorted.
    """
    suffix = metadata["suffix"]
    if part is None:
        return sorted(Path(source_dir).glob("**/[A-Za-z0-9]*" + suffix, recurse_symlinks=True))
    else:
        return sorted(Path(source_dir).glob(f"{part}/**/[A-Za-z0-9]*" + suffix, recurse_symlinks=True))


def get_subdirectories(directory: str | Path) -> list[Path]:
    """
    Returns all immediate subdirectories of the given directory (non-recursive)
    :param directory: Directory to search for subdirectories
    :return: List of Path objects to subdirectories. Sorted.
    """
    path = Path(directory)
    if not path.is_dir():
        raise NotADirectoryError(path)
    return sorted([entry for entry in path.iterdir() if entry.is_dir()])


class GenericJsonlReader:
    """
    Reader for jsonline files, compressed or not

    Following extensions and compressions are supported:
    * .jsonl.zst, .jsonl.zstd - ZStandard
    * .jsonl.gz - GZip
    * .jsonl - Uncompressed files (fallback)
    """

    def __init__(
        self,
        input_file_name: str | Path,
        encoding="utf-8",
        chunk_size: int = 16384,
        counter_name: str = "input",
    ):
        self._input_file_name = Path(input_file_name)
        self._chunk_size = chunk_size
        self._encoding = encoding
        self._counter_name = counter_name
        self._lines = 0

    def read(self) -> Generator[Any, Any, Iterator[Any] | None]:
        if not self._input_file_name.exists():
            logger.info(f"File not exist: {self._input_file_name}")
            return iter([])

        if self._input_file_name.suffix in [".zstd", ".zst"]:
            dctx = zstd.ZstdDecompressor()

            with open(self._input_file_name, "rb") as f:
                with dctx.stream_reader(f, read_size=self._chunk_size) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding=self._encoding)
                    for line in text_stream:
                        self._lines += 1
                        yield json.loads(line)
        elif self._input_file_name.suffix == ".gz":
            with gzip.open(self._input_file_name, "rb") as f:
                for line in f:
                    self._lines += 1
                    yield json.loads(line)
        else:
            with open(self._input_file_name, "rb") as f:
                for line in f:
                    self._lines += 1
                    yield json.loads(line)

    def get_metrics(self):
        return {
            self._counter_name: {
                "lines_read": self._lines,
            }
        }


class JsonlZstWriter:
    def __init__(
        self,
        output_file_name: str | Path,
        chunk_size: int = 16384,
        counter_name: str = "output",
    ):
        self._output_file_name = Path(output_file_name)
        self._chunk_size = chunk_size
        self._counter_name = counter_name
        self._lines = 0

    def write(self, iterator) -> None:
        cctx = zstd.ZstdCompressor()

        with open(self._output_file_name, "wb") as f:
            with cctx.stream_writer(f) as compressor:
                for item in iterator:
                    compressor.write(json.dumps(item))
                    compressor.write(b"\n")
                    self._lines += 1

    def get_metrics(self):
        return {
            self._counter_name: {
                "lines_written": self._lines,
            }
        }
