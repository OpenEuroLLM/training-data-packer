import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from training_data_packer.metadata import Metadata, get_megatron_lm_tokenizers, read_metadata
from training_data_packer.metadata.defaults import DEFAULT_BOS_TOKEN, DEFAULT_EOD_TOKEN, DEFAULT_SUFFIX, DEFAULT_TEXT
from training_data_packer.tokenizer.batching import iter_batches
from training_data_packer.tokenizer.indexed_dataset import DatasetWriter, dataset_exists, metrics_path, optimal_dtype
from training_data_packer.tokenizer.tokenization import GigatokenEncoder, TokenizerError
from training_data_packer.utils import slurm
from training_data_packer.utils.file import GenericJsonlReader, find_files
from training_data_packer.utils.metrics import write_metrics_to_file

DEFAULT_BATCH_BYTES = 256 * 1024 * 1024


def process(
    collection_dir: Path,
    args: dict[str, Any] | None = None,
) -> None:
    """Tokenize the collection with every tokenizer in the ``megatron-lm`` section.

    For each tokenizer, the files in its ``source`` directory are tokenized into
    ``<collection_dir>/megatron-lm/<tokenizer name>/``, keeping their relative paths.

    :param collection_dir: the collection directory holding ``metadata.yaml``
    :param args: command line options, e.g. ``workers`` and ``part``
    :raises ValueError: if no tokenizer is configured or a source directory has no files
    """
    if args is None:
        args = {}
    workers = args.get("workers", 1)
    part = args.get("part", None)
    metadata = read_metadata(collection_dir / "metadata.yaml")
    metadata["_internal"]["mode"] = "tokenize"

    tokenizers = get_megatron_lm_tokenizers(metadata)
    if len(tokenizers) == 0:
        raise ValueError("No tokenizer defined in section `megatron-lm`")

    tasks = []
    for name, config in tokenizers.items():
        source_dir = collection_dir / config["source"]
        src_files = find_files(source_dir, DEFAULT_SUFFIX, part)
        if len(src_files) == 0:
            logger.error(f"No files detected in {source_dir}. You have probably not run the packer in merge mode.")
            raise ValueError(f"No files detected in {source_dir}")
        logger.info(f"Found {len(src_files)} files in {source_dir} to tokenize with {name}")
        output_dir = collection_dir / "megatron-lm" / name
        tasks.extend(
            TokenizeTask(src_file, config["huggingface"], _output_file_prefix(src_file, source_dir, output_dir))
            for src_file in src_files
        )

    slurm.schedule_files(tasks, metadata, _run_task, workers, slurm.is_slurm_job())


@dataclass(frozen=True, slots=True)
class TokenizeTask:
    """A file to tokenize with one tokenizer."""

    src_file: Path
    tokenizer: str
    output_file_prefix: Path


def _run_task(task: TokenizeTask, _metadata: Metadata) -> None:
    tokenize_file(task.src_file, task.tokenizer, task.output_file_prefix)


def tokenize_file(
    src_file: Path,
    tokenizer: str,
    output_file_prefix: Path,
    text_field: str = DEFAULT_TEXT,
) -> None:
    """Tokenize a file into an indexed dataset.

    Encode the text field of every record in batches with the tokenizer, prepend the beginning-of-document token,
    append the end-of-document token, and write the tokens and a metrics file at ``output_file_prefix``. Skip the file
    if the output dataset already exists.

    :param src_file: the JSONL file to tokenize
    :param tokenizer: HuggingFace Hub repo id or local path of the tokenizer
    :param output_file_prefix: path prefix of the output dataset files
    :param text_field: the record field holding the text to tokenize
    :raises TokenizerError: if the tokenizer cannot be loaded or the beginning- or end-of-document token is not a
        single token
    :raises ValueError: if the beginning- or end-of-document token id is outside the tokenizer vocabulary
    :raises OSError: if reading the source file or writing the output fails
    """
    try:
        started = time.perf_counter()

        bos_token = DEFAULT_BOS_TOKEN
        eod_token = DEFAULT_EOD_TOKEN

        encoder = GigatokenEncoder.load(tokenizer)

        bos_id = encoder.token_to_id(bos_token)
        if bos_id is not None and not 0 <= bos_id < encoder.vocab_size:
            raise ValueError(f"bos id {bos_id} is outside the vocabulary (size {encoder.vocab_size})")

        eod_id = encoder.token_to_id(eod_token)
        if eod_id is not None and not 0 <= eod_id < encoder.vocab_size:
            raise ValueError(f"eod id {eod_id} is outside the vocabulary (size {encoder.vocab_size})")

        dtype = optimal_dtype(encoder.vocab_size)

        if dataset_exists(output_file_prefix):
            logger.info(f"Output files already exist {output_file_prefix}, skipping.")
            return

        metrics = Metrics(tokenizer, dtype=str(dtype), text=text_field, bos=bos_token, eod=eod_token)

        token_started = time.perf_counter()
        with DatasetWriter(output_file_prefix, dtype) as writer:
            src_reader = GenericJsonlReader(src_file)
            src_iter = src_reader.read()
            documents = map(lambda s: s[text_field], src_iter)
            for texts in iter_batches(documents, DEFAULT_BATCH_BYTES):
                batch = encoder.encode_batch(texts)
                if bos_id is not None:
                    batch = batch.with_bos(bos_id)
                if eod_id is not None:
                    batch = batch.with_eod(eod_id)
                writer.add_documents(batch.tokens, batch.lengths)
                metrics.documents += batch.num_documents
                metrics.empty_documents += batch.num_empty
                metrics.tokens += len(batch.tokens)
            writer.finalize()

            metrics.tokenize_seconds = time.perf_counter() - token_started
            metrics.total_seconds = time.perf_counter() - started
            write_metrics_to_file(dataclasses.asdict(metrics), metrics_path(output_file_prefix))

            logger.info(
                f"Wrote {metrics.documents:_} documents ({metrics.empty_documents:_} of them empty)"
                f" and {metrics.tokens:_} tokens in {metrics.total_seconds:.2f}s. Used tokenizer {metrics.tokenizer}.",
            )
    except (TokenizerError, OSError, ValueError) as exc:
        logger.error("%s", exc)
        raise exc


def _output_file_prefix(src_file: Path, source_dir: Path, output_dir: Path) -> Path:
    rel_path = src_file.relative_to(source_dir)
    return output_dir / rel_path.with_name(rel_path.name.removesuffix(DEFAULT_SUFFIX))


@dataclass(slots=True)
class Metrics:
    tokenizer: str
    dtype: str
    text: str
    bos: str
    eod: str
    documents: int = 0
    tokens: int = 0
    empty_documents: int = 0
    total_seconds: float = 0.0
    tokenize_seconds: float = 0.0
