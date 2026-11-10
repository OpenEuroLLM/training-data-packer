import io
import itertools
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import orjson as json
import zstandard as zstd
from loguru import logger

from training_data_packer.metadata import (
    Metadata,
    get_matching_part,
    get_shard_size_documents,
    read_metadata,
)
from training_data_packer.metadata.defaults import DEFAULT_SUFFIX, PREFIX_DEFAULT
from training_data_packer.utils import slurm
from training_data_packer.utils.file import find_files
from training_data_packer.utils.metrics import merge_metrics


def _shard_name(file_prefix: str, file_idx: int) -> str:
    """Build the file name of a merged shard.

    :param file_prefix: prefix of the shard file names
    :param file_idx: index of the shard
    :returns: the shard file name
    """
    return f"{file_prefix}_{file_idx:04d}{DEFAULT_SUFFIX}"


def _remove_unfinished_files(destination_dir: Path, file_prefix: str):
    """Remove hidden temporary shard files left behind by an interrupted merge.

    Only files belonging to ``file_prefix`` are removed, since other parts may write to the same directory.

    :param destination_dir: directory the shards are written to
    :param file_prefix: prefix of the shard file names
    """
    pattern = re.compile(rf"\.{re.escape(file_prefix)}_\d{{4,}}{re.escape(DEFAULT_SUFFIX)}")
    for path in destination_dir.iterdir():
        if path.is_file() and pattern.fullmatch(path.name):
            logger.info(f"Remove old temporary file {path}")
            path.unlink()


def _state_path(destination_dir: Path, file_prefix: str) -> Path:
    """Build the path of the hidden file holding the merge restart state.

    :param destination_dir: directory the shards are written to
    :param file_prefix: prefix of the shard file names
    :returns: path to the state file
    """
    return destination_dir.joinpath(f".{file_prefix}.merge-state.json")


def _write_state(state_path: Path, state: dict[str, Any]):
    """Atomically write the merge restart state.

    :param state_path: path to the state file
    :param state: the state to write
    """
    tmp_state_path = state_path.with_name(state_path.name + ".tmp")
    tmp_state_path.write_bytes(json.dumps(state))
    os.replace(tmp_state_path, state_path)


def _read_state(state_path: Path, input_files: list[Path], docs_per_shard: int) -> dict[str, Any] | None:
    """Read and validate the merge restart state.

    :param state_path: path to the state file
    :param input_files: files that are merged, in merge order
    :param docs_per_shard: maximum number of documents per shard
    :returns: the state, or None if there is no state to restart from
    :raises ValueError: if the state does not match the current input or shard size
    """
    if not state_path.exists():
        return None
    state = json.loads(state_path.read_bytes())
    if state["docs_per_shard"] != docs_per_shard:
        raise ValueError(
            f"Shard size changed from {state['docs_per_shard']} to {docs_per_shard} since {state_path} was written",
        )
    input_index = state["input_index"]
    if input_index >= len(input_files) or str(input_files[input_index]) != state["input_file"]:
        raise ValueError(f"Input files changed since {state_path} was written, cannot restart merge")
    return state


def merge(input_files: Iterable[Path], destination_dir: Path, docs_per_shard: int, file_prefix: str):
    """Merge input files into shards of ``docs_per_shard`` documents.

    Each shard is written to a hidden temporary file and renamed to its final name once complete.
    When a shard starts, its position in the input is saved to a hidden state file. If a previous merge
    was interrupted, the merge restarts from the saved position of the last started shard.

    :param input_files: zstd compressed JSON Lines files to merge
    :param destination_dir: directory to write the shards to
    :param docs_per_shard: maximum number of documents per shard
    :param file_prefix: prefix of the shard file names
    :raises FileExistsError: if a shard with the same name already exists
    """
    logger.info(f"Writing to directory {destination_dir}, using prefix {file_prefix}")
    os.makedirs(destination_dir, exist_ok=True)
    _remove_unfinished_files(destination_dir, file_prefix)

    input_files = list(input_files)
    state_path = _state_path(destination_dir, file_prefix)
    state = _read_state(state_path, input_files, docs_per_shard)
    if state is None:
        file_idx, start_input, start_line = 0, 0, 0
    else:
        file_idx, start_input, start_line = state["shard"], state["input_index"], state["line"]
        logger.info(f"Restart merge at shard {file_idx} from line {start_line} of {state['input_file']}")
        # The shard may have been renamed before the state of the next shard was saved, so redo it
        done_shard = destination_dir.joinpath(_shard_name(file_prefix, file_idx))
        if done_shard.exists():
            logger.info(f"Remove {done_shard} to redo it")
            done_shard.unlink()

    dctx = zstd.ZstdDecompressor()
    cctx = zstd.ZstdCompressor(level=3)

    docs_written = 0
    out_f = None
    writer = None
    output_path = None
    tmp_output_path = None

    try:
        for input_idx in range(start_input, len(input_files)):
            file_path = input_files[input_idx]
            first_line = start_line if input_idx == start_input else 0
            with open(file_path, "rb") as in_f:
                with dctx.stream_reader(in_f) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                    lines = itertools.islice(text_stream, first_line, None)
                    for line_idx, line in enumerate(lines, start=first_line):
                        if out_f is None or docs_written >= docs_per_shard:
                            if out_f:
                                writer.close()
                                out_f.close()
                                os.rename(tmp_output_path, output_path)
                                out_f = None

                            output_path = destination_dir.joinpath(_shard_name(file_prefix, file_idx))
                            tmp_output_path = destination_dir.joinpath("." + output_path.name)
                            if output_path.exists():
                                logger.warning(f"File {output_path} already exists, aborting part merge")
                                raise FileExistsError(f"File {output_path} already exists, aborting part merge")
                            _write_state(
                                state_path,
                                {
                                    "shard": file_idx,
                                    "input_index": input_idx,
                                    "input_file": str(file_path),
                                    "line": line_idx,
                                    "docs_per_shard": docs_per_shard,
                                },
                            )
                            out_f = open(tmp_output_path, "wb")
                            writer = cctx.stream_writer(out_f)

                            docs_written = 0
                            file_idx += 1
                        writer.write(line.encode("utf-8"))
                        docs_written += 1
    finally:
        if out_f:
            writer.close()
            out_f.close()
    if out_f:
        os.rename(tmp_output_path, output_path)
    state_path.unlink(missing_ok=True)


def merge_part(part_name: str, metadata: Metadata):
    collection_dir = metadata.get("_internal.collection_dir")
    input_dir = collection_dir.joinpath("release-raw")
    output_dir = collection_dir.joinpath("release")
    part_config, _ = get_matching_part(metadata, part_name)
    if part_config is None:
        logger.error(f"Could not find config for part {part_name}")
        raise ValueError(f"Could not find config for part {part_name}")
    logger.info(f"Processing part {part_name} with config {part_config}")
    flat_output = part_config["pack"] == "flat"
    metadata["suffix"] = DEFAULT_SUFFIX
    files = find_files(input_dir.joinpath(part_name), metadata.get("suffix", DEFAULT_SUFFIX))
    logger.info(f"Processing part {part_name} with {len(files)} files")
    docs_per_shard = get_shard_size_documents(part_config)
    merge(
        files,
        output_dir if flat_output else output_dir.joinpath(part_name),
        docs_per_shard,
        part_config.get("prefix", PREFIX_DEFAULT),
    )
    if not flat_output:
        merge_metrics(input_dir.joinpath(part_name), output_dir.joinpath(part_name))


def is_single_task_merge(metadata: Metadata, parts: list[str]) -> bool:
    """Check if all parts are flattened into one, which must run in a single task.

    :param metadata: metadata of the collection
    :param parts: names of the parts to merge
    :returns: True if the merge must run single threaded in one task
    """
    return metadata.get("release.default.pack") == "flat" and metadata.get(f"release.'{parts[0]}'.prefix") is None


def process(
    collection_dir: Path,
    args: dict[str, Any] | None = None,
) -> bool:
    if args is None:
        args = {}
    workers = args.get("workers", 1)
    part = args.get("part", None)
    metadata = read_metadata(collection_dir.joinpath("metadata.yaml"))
    metadata["_internal"]["mode"] = "merge"
    input_dir = collection_dir.joinpath("release-raw")
    output_dir = collection_dir.joinpath("release")

    if part is not None:
        parts = [part]
    else:
        parts = metadata.get_all_part_names("release")
    logger.info(f"Found {len(parts)} parts")

    pack_method = metadata.get("release.default.pack")
    if pack_method is None:
        logger.error("Pack method not set, abort")
        return False

    if is_single_task_merge(metadata, parts):
        # This config requires single threaded
        workers = 1
        parts = ["default"]
        logger.info("All parts will be flatten into one. This will run single threaded.")

    if parts == ["default"]:
        if (slurm.is_slurm_job() and slurm.get_array_task_id() == 0) or not slurm.is_slurm_job():
            metadata["suffix"] = DEFAULT_SUFFIX
            files = find_files(input_dir, metadata.get("suffix", DEFAULT_SUFFIX))
            docs_per_shard = get_shard_size_documents(metadata["release.default"])
            merge(
                files,
                output_dir,
                docs_per_shard,
                PREFIX_DEFAULT,
            )
            merge_metrics(input_dir, output_dir)
        else:
            logger.info(f"Only one task is allowed for this merge. My task id is {slurm.get_array_task_id()}")
        return True
    else:
        try:
            slurm.schedule_files(parts, metadata, merge_part, workers, slurm.is_slurm_job())
            if not slurm.is_slurm_job() or (slurm.is_slurm_job() and slurm.get_array_task_id() == 0):
                merge_metrics(input_dir, output_dir)
            return True
        except RuntimeError:
            return False
