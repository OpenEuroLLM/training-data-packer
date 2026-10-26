"""Shared test fixtures and helpers."""

from pathlib import Path

import pytest

from tests.helpers.tokenizer import EOS, EOT, write_tiny_tokenizer

__all__ = ["EOS", "EOT"]


@pytest.fixture(scope="session")
def tiny_tokenizer_json(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Minimal byte-level BPE tokenizer.json.

    256 byte tokens, one merge ("ab"), EOT and EOS.
    """
    return write_tiny_tokenizer(tmp_path_factory.mktemp("tokenizer") / "tokenizer.json")
