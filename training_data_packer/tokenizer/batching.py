"""Group documents into size-bounded batches for tokenization."""

from collections.abc import Iterable, Iterator


def iter_batches(documents: Iterable[str], max_batch_bytes: int) -> Iterator[list[str]]:
    """Group documents into batches of roughly ``max_batch_bytes`` each.

    The size of a batch is the sum of ``len()`` of its documents, so it is measured in characters
    (Unicode code points), not encoded bytes. Documents are added to the current batch in input order,
    and the batch is yielded as soon as its size reaches ``max_batch_bytes``. A batch can therefore
    exceed the budget, but only because of its last document, and a single document larger than the
    budget closes the batch it is added to. The final batch holds any remaining documents and can be
    smaller than the budget.

    :param documents: the documents to batch, consumed lazily
    :param max_batch_bytes: the size at which a batch is yielded, in characters
    :returns: an iterator over non-empty batches that together contain every document in input order
    :raises ValueError: if ``max_batch_bytes`` is not positive
    """
    if max_batch_bytes <= 0:
        raise ValueError("max_batch_bytes must be positive")
    batch: list[str] = []
    batch_size = 0
    for doc in documents:
        batch.append(doc)
        batch_size += len(doc)
        if batch_size >= max_batch_bytes:
            yield batch
            batch, batch_size = [], 0
    if batch:
        yield batch
