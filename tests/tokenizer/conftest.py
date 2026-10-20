"""Shared test fixtures and helpers."""

import json
from pathlib import Path

import pytest

EOT = "<|endoftext|>"
EOS = "<eos>"


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte -> printable unicode character mapping."""
    printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    chars = printable[:]
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            chars.append(256 + extra)
            extra += 1
    return {b: chr(c) for b, c in zip(printable, chars, strict=True)}


@pytest.fixture(scope="session")
def tiny_tokenizer_json(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Minimal byte-level BPE tokenizer.json.

    256 byte tokens, one merge ("ab"), EOT and EOS.
    """
    b2u = _bytes_to_unicode()
    vocab = {b2u[b]: b for b in range(256)}
    vocab[b2u[ord("a")] + b2u[ord("b")]] = 256
    config = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": 257,
                "content": EOT,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
            {
                "id": 258,
                "content": EOS,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": [[b2u[ord("a")], b2u[ord("b")]]],
        },
    }
    path = tmp_path_factory.mktemp("tokenizer") / "tokenizer.json"
    path.write_text(json.dumps(config))
    return path
