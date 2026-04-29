import io
import json
from collections.abc import Generator, Iterator
from os import PathLike
from pathlib import Path
from typing import Any

import zstandard as zstd
from loguru import logger


class JsonlZstReader:
    def __init__(self, input_file_name: str | PathLike[str], encoding='utf-8', chunk_size: int=16384):
        self._input_file_name = Path(input_file_name)
        self._chunk_size = chunk_size
        self._encoding = encoding

    def read(self) -> Generator[Any, Any, Iterator[Any] | None]:
        if not self._input_file_name.exists():
            logger.info(f"File not exist: {self._input_file_name}")
            return iter([])

        dctx = zstd.ZstdDecompressor()

        with open(self._input_file_name, 'rb') as f:
            with dctx.stream_reader(f, read_size=self._chunk_size) as reader:
                text_stream = io.TextIOWrapper(reader, encoding=self._encoding)
                for line in text_stream:
                    yield json.loads(line)


class JsonlZstWriter:
    def __init__(self, output_file_name: str | PathLike[str], encoding='utf-8', chunk_size: int=16384):
        self._output_file_name = Path(output_file_name)
        self._chunk_size = chunk_size
        self._encoding = encoding

    def write(self, iterator) -> None:
        cctx = zstd.ZstdCompressor()

        with open(self._output_file_name, 'wb') as f:
            with cctx.stream_writer(f) as compressor:
                for item in iterator:
                    compressor.write(json.dumps(item).encode(self._encoding))
                    compressor.write(b'\n')



