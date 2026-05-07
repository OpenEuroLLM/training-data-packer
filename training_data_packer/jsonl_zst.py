from pathlib import Path

import orjson as json
import zstandard as zstd


class JsonlZstWriter:
    def __init__(
        self,
        output_file_name: str | Path,
        chunk_size: int = 16384,
    ):
        self._output_file_name = Path(output_file_name)
        self._chunk_size = chunk_size

    def write(self, iterator) -> None:
        cctx = zstd.ZstdCompressor()

        with open(self._output_file_name, "wb") as f:
            with cctx.stream_writer(f) as compressor:
                for item in iterator:
                    compressor.write(json.dumps(item))
                    compressor.write(b"\n")
