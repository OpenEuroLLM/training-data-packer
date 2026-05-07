import io
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any

import orjson as json
import zstandard as zstd
from loguru import logger


def find_jsonl_zst_files(source_dir: Path, release: str = None) -> list[Path]:
    if release is None:
        return sorted(Path(source_dir).glob("**/[A-Za-z0-9]*.jsonl.zst"))
    else:
        return sorted(Path(source_dir).glob(f"{release}/**/[A-Za-z0-9]*.jsonl.zst"))


def get_directories_n_levels_down(base_path: Path, n_levels: int) -> list[Path]:
    glob = "/".join(["*"] * n_levels)
    dirs = [p for p in base_path.glob(glob) if p.is_dir()]
    return dirs


class GenericJsonlReader:
    def __init__(
        self,
        input_file_name: str | Path,
        encoding="utf-8",
        chunk_size: int = 16384,
    ):
        self._input_file_name = Path(input_file_name)
        self._compressed = str(input_file_name).endswith(".zstd") or str(input_file_name).endswith(".zst")
        self._chunk_size = chunk_size
        self._encoding = encoding

    def read(self) -> Generator[Any, Any, Iterator[Any] | None]:
        if not self._input_file_name.exists():
            logger.info(f"File not exist: {self._input_file_name}")
            return iter([])

        if self._compressed:
            dctx = zstd.ZstdDecompressor()

            with open(self._input_file_name, "rb") as f:
                with dctx.stream_reader(f, read_size=self._chunk_size) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding=self._encoding)
                    for line in text_stream:
                        yield json.loads(line)
        else:
            with open(self._input_file_name, "rb") as f:
                for line in f:
                    yield json.loads(line)
