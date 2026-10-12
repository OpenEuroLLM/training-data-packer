import gzip
import io
import re
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any

import orjson as json
import pyarrow.parquet as pq
import zstandard as zstd
from loguru import logger


def find_files(source_dir: Path, suffix: str, part: str | None = None) -> list[Path]:
    """Return all files matching suffix criteria.

    Return all non-hidden files under source_dir, following symlinks.

    :param source_dir: directory to find files in
    :param suffix: expected file suffix
    :param part: optional part, if only a part of source_dir is needed
    :returns: sorted list of paths to the files
    """
    if part is None:
        return sorted(Path(source_dir).glob("**/[A-Za-z0-9]*" + suffix, recurse_symlinks=True))
    else:
        return sorted(Path(source_dir).glob(f"{part}/**/[A-Za-z0-9]*" + suffix, recurse_symlinks=True))


def get_subdirectories(directory: str | Path) -> list[Path]:
    """Get all immediate subdirectories.

    Return all immediate subdirectories of the given directory (non-recursive).

    :param directory: directory to search for subdirectories
    :returns: sorted list of paths to the subdirectories
    """
    path = Path(directory)
    if not path.is_dir():
        raise NotADirectoryError(path)
    return sorted([entry for entry in path.iterdir() if entry.is_dir()])


def change_suffix(filename: str | Path, original_suffix, new_suffix) -> Path:
    if original_suffix == new_suffix:
        return Path(filename)
    new_filename = re.sub(f"(.*){original_suffix}", f"\\1{new_suffix}", str(filename))
    return Path(new_filename)


def prepare_output_file(out_file: Path) -> tuple[Path, bool]:
    """Prepare output file path and check if processing should continue.

    Create the temporary file path, clean up old temporary files, and ensure the
    output directory exists.

    :param out_file: target output file path
    :returns: tuple of the temporary file path and whether processing should continue
    """
    tmp_out_file = out_file.parent.joinpath("." + out_file.name)
    if out_file.exists():
        return tmp_out_file, False
    if tmp_out_file.exists():
        logger.info(f"Remove old temporary file {tmp_out_file}")
        tmp_out_file.unlink()
    out_file.parent.mkdir(exist_ok=True, parents=True)
    return tmp_out_file, True


class GenericJsonlReader:
    """Read JSON Lines files, compressed or not.

    The following extensions and compressions are supported:

    * .jsonl.zst, .jsonl.zstd - ZStandard
    * .jsonl.gz - GZip
    * .parquet - files in parquet format
    * .jsonl - uncompressed files (fallback)

    :param input_file_name: path to the file to read
    :param encoding: text encoding of the file
    :param chunk_size: size of the chunks read from the file
    :param counter_name: key used for the metrics
    """

    def __init__(
        self,
        input_file_name: str | Path | None,
        encoding="utf-8",
        chunk_size: int = 16384,
        counter_name: str = "input",
    ):
        if input_file_name is None or input_file_name == "":
            self._input_file_name = None
        else:
            self._input_file_name = Path(input_file_name)
        self._chunk_size = chunk_size
        self._encoding = encoding
        self._counter_name = counter_name
        self._lines = 0

    def read(self) -> Generator[Any, Any, Iterator[Any] | None]:
        if self._input_file_name is None or not self._input_file_name.exists():
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
        elif self._input_file_name.suffix == ".parquet":
            parquet_file = pq.ParquetFile(self._input_file_name)
            for batch in parquet_file.iter_batches():
                for line in batch.to_pylist():
                    self._lines += 1
                    yield line
        else:
            with open(self._input_file_name, "rb") as f:
                for line in f:
                    self._lines += 1
                    yield json.loads(line)

    def get_metrics(self):
        return {
            self._counter_name: {
                "document_read": self._lines,
            }
        }


class JsonlZstWriter:
    """Write JSON Lines compressed as ZStandard.

    :param output_file_name: path to the file to write
    :param chunk_size: size of the chunks written to the file
    :param counter_name: key used for the metrics
    """

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
        self._size = 0

    def write(self, iterator) -> None:
        cctx = zstd.ZstdCompressor()

        with open(self._output_file_name, "wb") as f:
            with cctx.stream_writer(f) as compressor:
                for item in iterator:
                    compressor.write(json.dumps(item))
                    compressor.write(b"\n")
                    self._lines += 1
        self._size = self._output_file_name.stat().st_size

    def get_metrics(self):
        return {
            self._counter_name: {
                "document_written": self._lines,
                "size_bytes": self._size,
            }
        }
