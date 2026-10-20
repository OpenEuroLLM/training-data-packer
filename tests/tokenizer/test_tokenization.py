from pathlib import Path

import numpy as np
import pytest

from tests.tokenizer.conftest import EOS, EOT
from training_data_packer.tokenizer.tokenization import EncodedBatch, GigatokenEncoder, TokenizerError


def batch(*documents: list[int]) -> EncodedBatch:
    tokens = np.array([t for d in documents for t in d], dtype=np.uint32)
    return EncodedBatch(tokens=tokens, lengths=np.array([len(d) for d in documents], dtype=np.int64))


def split(encoded: EncodedBatch) -> list[list[int]]:
    return [part.tolist() for part in np.split(encoded.tokens, np.cumsum(encoded.lengths)[:-1])]


class TestEncodedBatch:
    def test_rejects_inconsistent_lengths(self) -> None:
        with pytest.raises(ValueError, match="lengths"):
            EncodedBatch(tokens=np.array([1, 2]), lengths=np.array([1]))

    def test_num_empty(self) -> None:
        result = batch([], [1, 2], [], [3], [])
        assert result.num_documents == 5
        assert result.num_empty == 3

    def test_with_eod_appends_to_every_document(self) -> None:
        result = batch([1, 2], [3], [4, 5, 6]).with_eod(9)
        assert split(result) == [[1, 2, 9], [3, 9], [4, 5, 6, 9]]
        assert result.tokens.dtype == np.uint32

    def test_with_eod_leaves_empty_documents_empty(self) -> None:
        result = batch([], [1], [], [2, 3], []).with_eod(9)
        assert result.lengths.tolist() == [0, 2, 0, 3, 0]
        assert result.tokens.tolist() == [1, 9, 2, 3, 9]


class TestGigatokenEncoder:
    @pytest.fixture
    def encoder(self, tiny_tokenizer_json: Path) -> GigatokenEncoder:
        return GigatokenEncoder.load(str(tiny_tokenizer_json))

    def test_vocab_size_and_fingerprint(self, encoder: GigatokenEncoder, tiny_tokenizer_json: Path) -> None:
        assert encoder.vocab_size == 259
        assert str(tiny_tokenizer_json) in encoder.fingerprint

    def test_encode_batch_matches_reference_tokenization(self, encoder: GigatokenEncoder) -> None:
        result = encoder.encode_batch(["ab c", "", "hello ab", "åäö"])
        expected_ab_c = [256, ord(" "), ord("c")]
        expected_hello = [*b"hello", ord(" "), 256]
        assert split(result) == [expected_ab_c, [], expected_hello, list("åäö".encode())]

    def test_encode_empty_batch(self, encoder: GigatokenEncoder) -> None:
        result = encoder.encode_batch([])
        assert result.num_documents == 0
        assert len(result.tokens) == 0

    def test_token_to_id_for_special_token(self, encoder: GigatokenEncoder) -> None:
        assert encoder.token_to_id(EOT) == 257
        assert encoder.token_to_id(EOS) == 258

    def test_token_to_id_rejects_multi_token_strings(self, encoder: GigatokenEncoder) -> None:
        with pytest.raises(TokenizerError, match="not a single token"):
            encoder.token_to_id("</s>")

    def test_load_failure_raises_tokenizer_error(self, tmp_path: Path) -> None:
        with pytest.raises(TokenizerError, match="could not load tokenizer"):
            GigatokenEncoder.load(str(tmp_path / "missing" / "tokenizer.json"))


@pytest.mark.network
def test_real_hub_tokenizer() -> None:
    encoder = GigatokenEncoder.load("openai-community/gpt2")
    assert encoder.vocab_size == 50257
    assert encoder.token_to_id(EOT) == 50256
    assert split(encoder.encode_batch(["Hello world"])) == [[15496, 995]]


@pytest.mark.network
def test_default_openeurollm_tokenizer() -> None:
    encoder = GigatokenEncoder.load("openeurollm/tokenizer-256k-v2")
    assert encoder.vocab_size == 262144
    assert encoder.token_to_id("<eos>") == 2
