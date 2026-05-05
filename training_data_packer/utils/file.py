from pathlib import Path


def find_jsonl_zst_files(source_dir: Path, release: str = None) -> list[Path]:
    if release is None:
        return sorted(Path(source_dir).glob("**/[A-Za-z0-9]*.jsonl.zst"))
    else:
        return sorted(Path(source_dir).glob(f"{release}/**/[A-Za-z0-9]*.jsonl.zst"))


def get_directories_n_levels_down(base_path: Path, n_levels: int) -> list[Path]:
    glob = "/".join(["*"] * n_levels)
    dirs = [p for p in base_path.glob(glob) if p.is_dir()]
    return dirs
