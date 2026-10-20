import pytest

from training_data_packer.tokenizer.batching import iter_batches


class TestIterBatches:
    def test_batches_respect_budget(self) -> None:
        batches = list(iter_batches(["aa", "bb", "cc", "dd", "e"], max_batch_bytes=4))
        assert batches == [["aa", "bb"], ["cc", "dd"], ["e"]]

    def test_oversized_document_forms_its_own_batch(self) -> None:
        assert list(iter_batches(["a", "x" * 100, "b"], max_batch_bytes=10)) == [["a", "x" * 100], ["b"]]

    def test_empty_input(self) -> None:
        assert list(iter_batches([], max_batch_bytes=10)) == []

    def test_preserves_all_documents_in_order(self) -> None:
        docs = [str(i) * (i % 7) for i in range(100)]
        batches = list(iter_batches(docs, max_batch_bytes=13))
        assert [d for b in batches for d in b] == docs

    def test_rejects_non_positive_budget(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            list(iter_batches(["a"], max_batch_bytes=0))
