from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ByteTokenizer:
    """Deterministic UTF-8 byte tokenizer with no learned vocabulary."""

    vocab_size = 256

    @staticmethod
    def encode(text: str) -> list[int]:
        return list(text.encode("utf-8"))

    @staticmethod
    def decode(token_ids: list[int]) -> str:
        return bytes(token_ids).decode("utf-8", errors="replace")


class TokenBlockDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, token_path: Path, sequence_length: int) -> None:
        # The configured corpora are deliberately small (a few MB). Loading the
        # array avoids lingering Windows mmap handles and keeps worker behavior
        # straightforward and reproducible.
        self.tokens = np.load(token_path, allow_pickle=False)
        self.sequence_length = sequence_length
        self.num_blocks = (len(self.tokens) - 1) // sequence_length
        if self.num_blocks < 1:
            raise ValueError(
                f"{token_path} has too few tokens for sequence length {sequence_length}"
            )

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.sequence_length
        block = np.asarray(
            self.tokens[start : start + self.sequence_length + 1],
            dtype=np.int64,
        ).copy()
        values = torch.from_numpy(block)
        return values[:-1], values[1:]


def _encode_split(dataset: object, max_tokens: int | None) -> np.ndarray:
    token_buffer = bytearray()
    for row in dataset:
        text = row["text"]
        if not text.strip():
            continue
        token_buffer.extend(text.encode("utf-8"))
        token_buffer.extend(b"\n")
        if max_tokens and len(token_buffer) >= max_tokens:
            del token_buffer[max_tokens:]
            break
    return np.frombuffer(bytes(token_buffer), dtype=np.uint8)


def prepare_wikitext2(
    cache_dir: Path,
    max_train_tokens: int | None,
    max_validation_tokens: int | None,
) -> dict[str, Path]:
    """Download WikiText-2 and cache deterministic byte-token arrays locally."""
    from datasets import load_dataset

    cache_dir.mkdir(parents=True, exist_ok=True)
    limits = {
        "train": max_train_tokens,
        "validation": max_validation_tokens,
    }
    paths: dict[str, Path] = {}
    metadata: dict[str, object] = {
        "dataset": "Salesforce/wikitext",
        "subset": "wikitext-2-raw-v1",
        "tokenizer": "utf-8-bytes",
        "vocab_size": ByteTokenizer.vocab_size,
        "splits": {},
    }
    for split, limit in limits.items():
        suffix = "all" if limit is None else str(limit)
        token_path = cache_dir / f"wikitext2_{split}_{suffix}_bytes.npy"
        if not token_path.exists():
            raw = load_dataset(
                "Salesforce/wikitext",
                "wikitext-2-raw-v1",
                split=split,
                cache_dir=str(cache_dir / "huggingface"),
            )
            tokens = _encode_split(raw, limit)
            np.save(token_path, tokens, allow_pickle=False)
        token_count = int(np.load(token_path, mmap_mode="r").shape[0])
        metadata["splits"][split] = {
            "path": str(token_path),
            "tokens": token_count,
        }
        paths[split] = token_path
    (cache_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return paths
