r"""Reading, writing and merging Megatron-LM ``MMapIndexedDataset`` (bin/idx) files.

A dataset with prefix ``foo`` consists of two files:

* ``foo.bin`` - all token ids, concatenated, in a fixed numpy dtype.
* ``foo.idx`` - an index with the layout used by
  ``megatron.core.datasets.indexed_dataset``::

      magic            9 bytes   b"MMIDIDX\\x00\\x00"
      version          uint64    1
      dtype code       uint8     see ``_DTYPE_CODES``
      sequence count   uint64    N
      document count   uint64    D (number of documents + 1)
      sequence_lengths int32[N]  tokens per sequence
      sequence_pointers int64[N] byte offset of each sequence in the .bin file
      document_indices int64[D]  sequence index where each document starts, plus the end

All integers are little-endian. Like Megatron's ``preprocess_data.py``, this tool
writes every document as a single sequence, except that a document without tokens
gets no sequence but still an entry in ``document_indices``. Documents
``[a], [], [b]`` therefore give ``document_indices = [0, 1, 1, 2]``.
"""

import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np
import numpy.typing as npt

INDEX_HEADER = b"MMIDIDX\x00\x00"
INDEX_VERSION = 1

_DTYPE_CODES: dict[np.dtype, int] = {
    np.dtype(np.uint8): 1,
    np.dtype(np.int8): 2,
    np.dtype(np.int16): 3,
    np.dtype(np.int32): 4,
    np.dtype(np.int64): 5,
    np.dtype(np.float64): 6,
    np.dtype(np.float32): 7,
    np.dtype(np.uint16): 8,
}
_CODE_DTYPES = {code: dtype for dtype, code in _DTYPE_CODES.items()}

# Matches megatron.core.datasets.indexed_dataset.DType.optimal_dtype.
_UINT16_VOCAB_LIMIT = 65500


class IndexFormatError(ValueError):
    """Raised when an .idx file is not a valid Megatron index."""


def optimal_dtype(vocab_size: int) -> np.dtype:
    """Compute smallest token dtype.

    :vocab_size: Vocabulary size.
    :returns: The smallest token dtype Megatron uses for a vocabulary of ``vocab_size``.
    """
    return np.dtype(np.uint16) if vocab_size < _UINT16_VOCAB_LIMIT else np.dtype(np.int32)


def bin_path(prefix: Path) -> Path:
    return prefix.with_name(prefix.name + ".bin")


def idx_path(prefix: Path) -> Path:
    return prefix.with_name(prefix.name + ".idx")


def metrics_path(prefix: Path) -> Path:
    return prefix.with_name(prefix.name + ".json")


def dataset_exists(prefix: Path) -> bool:
    """Whether a complete dataset exists at ``prefix``.

    Writers only move the .idx file into place after the .bin file, so the presence
    of both means the dataset was finalized.
    """
    return bin_path(prefix).is_file() and idx_path(prefix).is_file() and metrics_path(prefix).is_file()


def _tmp(path: Path) -> Path:
    return path.with_name("." + path.name)


@dataclass(frozen=True, slots=True)
class DatasetIndex:
    """The in-memory contents of an .idx file."""

    dtype: np.dtype
    sequence_lengths: npt.NDArray[np.int32]
    document_indices: npt.NDArray[np.int64]

    @classmethod
    def from_document_lengths(cls, dtype: np.dtype, document_lengths: npt.ArrayLike) -> Self:
        """Build an index with one sequence per document, or none for a document without tokens."""
        lengths = np.asarray(document_lengths, dtype=np.int32)
        non_empty = lengths > 0
        document_indices = np.zeros(len(lengths) + 1, dtype=np.int64)
        np.cumsum(non_empty, dtype=np.int64, out=document_indices[1:])
        return cls(dtype=np.dtype(dtype), sequence_lengths=lengths[non_empty], document_indices=document_indices)

    @property
    def num_sequences(self) -> int:
        return len(self.sequence_lengths)

    @property
    def num_tokens(self) -> int:
        return int(self.sequence_lengths.sum(dtype=np.int64))

    @property
    def sequence_pointers(self) -> npt.NDArray[np.int64]:
        """Byte offset of each sequence in the .bin file."""
        offsets = np.zeros(self.num_sequences, dtype=np.int64)
        np.cumsum(self.sequence_lengths[:-1], dtype=np.int64, out=offsets[1:])
        return offsets * self.dtype.itemsize

    def write(self, path: Path) -> None:
        if self.dtype not in _DTYPE_CODES:
            raise ValueError(f"unsupported dtype for Megatron index: {self.dtype}")
        with path.open("wb") as f:
            f.write(INDEX_HEADER)
            f.write(struct.pack("<Q", INDEX_VERSION))
            f.write(struct.pack("<B", _DTYPE_CODES[self.dtype]))
            f.write(struct.pack("<Q", self.num_sequences))
            f.write(struct.pack("<Q", len(self.document_indices)))
            f.write(self.sequence_lengths.astype("<i4", copy=False).tobytes(order="C"))
            f.write(self.sequence_pointers.astype("<i8", copy=False).tobytes(order="C"))
            f.write(self.document_indices.astype("<i8", copy=False).tobytes(order="C"))

    @classmethod
    def read(cls, path: Path) -> Self:
        data = path.read_bytes()
        header_size = len(INDEX_HEADER) + 8 + 1 + 8 + 8
        if len(data) < header_size or not data.startswith(INDEX_HEADER):
            raise IndexFormatError(f"{path}: not a Megatron index file")
        offset = len(INDEX_HEADER)
        (version,) = struct.unpack_from("<Q", data, offset)
        if version != INDEX_VERSION:
            raise IndexFormatError(f"{path}: unsupported index version {version}")
        (code,) = struct.unpack_from("<B", data, offset + 8)
        if code not in _CODE_DTYPES:
            raise IndexFormatError(f"{path}: unknown dtype code {code}")
        sequence_count, document_count = struct.unpack_from("<QQ", data, offset + 9)
        offset = header_size
        expected_size = offset + sequence_count * (4 + 8) + document_count * 8
        if len(data) != expected_size:
            raise IndexFormatError(f"{path}: expected {expected_size} bytes, found {len(data)}")
        lengths = np.frombuffer(data, dtype="<i4", count=sequence_count, offset=offset)
        offset += sequence_count * 4
        pointers = np.frombuffer(data, dtype="<i8", count=sequence_count, offset=offset)
        offset += sequence_count * 8
        documents = np.frombuffer(data, dtype="<i8", count=document_count, offset=offset)
        index = cls(
            dtype=_CODE_DTYPES[code],
            sequence_lengths=lengths.astype(np.int32),
            document_indices=documents.astype(np.int64),
        )
        if not np.array_equal(index.sequence_pointers, pointers):
            raise IndexFormatError(f"{path}: sequences are not stored contiguously")
        return index


class DatasetWriter:
    """Streams documents into a new bin/idx dataset.

    Data goes to temporary files that are moved into place by :meth:`finalize`, so an
    interrupted run never leaves a dataset that looks complete. Use it as a context
    manager to have the temporary files removed if an exception occurs.
    """

    def __init__(self, prefix: Path, dtype: np.dtype) -> None:
        if np.dtype(dtype) not in _DTYPE_CODES:
            raise ValueError(f"unsupported dtype for Megatron index: {dtype}")
        self.prefix = prefix
        self.dtype = np.dtype(dtype)
        self._max_value = np.iinfo(self.dtype).max
        self._lengths: list[npt.NDArray[np.int32]] = []
        prefix.parent.mkdir(parents=True, exist_ok=True)
        self._bin_tmp = _tmp(bin_path(prefix))
        self._idx_tmp = _tmp(idx_path(prefix))
        self._bin = self._bin_tmp.open("wb")

    def add_documents(self, tokens: npt.NDArray[np.integer], lengths: npt.NDArray[np.integer]) -> None:
        """Append documents given as one flat token array plus the length of each document.

        A document of length 0 is recorded in the index without a sequence.
        """
        lengths = np.asarray(lengths)
        if int(lengths.sum()) != len(tokens):
            raise ValueError("sum of lengths does not match the number of tokens")
        if len(tokens) and int(tokens.max()) > self._max_value:
            raise ValueError(f"token id {int(tokens.max())} does not fit in {self.dtype}")
        self._bin.write(np.ascontiguousarray(tokens, dtype=self.dtype).tobytes())
        self._lengths.append(lengths.astype(np.int32))

    def finalize(self) -> DatasetIndex:
        """Write the index and move both files into their final location."""
        self._bin.close()
        lengths = np.concatenate(self._lengths) if self._lengths else np.empty(0, np.int32)
        index = DatasetIndex.from_document_lengths(self.dtype, lengths)
        index.write(self._idx_tmp)
        self._bin_tmp.replace(bin_path(self.prefix))
        self._idx_tmp.replace(idx_path(self.prefix))
        return index

    def abort(self) -> None:
        """Discard everything written so far."""
        self._bin.close()
        self._bin_tmp.unlink(missing_ok=True)
        self._idx_tmp.unlink(missing_ok=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            self.abort()


def read_documents(prefix: Path) -> Iterator[npt.NDArray[np.integer]]:
    """Yield the token ids of every document in a dataset (memory-mapped)."""
    index = DatasetIndex.read(idx_path(prefix))
    if index.num_sequences == 0:
        return
    tokens = np.memmap(bin_path(prefix), dtype=index.dtype, mode="r")
    starts = index.sequence_pointers // index.dtype.itemsize
    for start, length in zip(starts, index.sequence_lengths, strict=True):
        yield tokens[start : start + length]
