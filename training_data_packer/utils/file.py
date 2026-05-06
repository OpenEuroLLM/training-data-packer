import json
from pathlib import Path
from typing import Generator, Any, Iterator

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


class JsonlReader:
    def __init__(
        self,
        input_file_name: str | Path,
    ):
        self._input_file_name = Path(input_file_name)

    def read(self) -> Generator[Any, Any, Iterator[Any] | None]:
        if not self._input_file_name.exists():
            logger.info(f"File not exist: {self._input_file_name}")
            return iter([])

        with open(self._input_file_name, "rb") as f:
            for line in f:
                yield json.loads(line)
