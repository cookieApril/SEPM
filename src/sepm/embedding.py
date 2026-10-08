"""Deterministic cross-process text embeddings without external model dependencies.

HashingEmbedder supports local tests and paper baselines. It is fast and needs
no downloaded model, but it is not a production semantic embedder. The
``Embedder`` protocol lets callers inject a real vector model.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol


class Embedder(Protocol):
    """Minimal interface implemented by every embedding backend."""
    def embed(self, text: str) -> list[float]: ...


class HashingEmbedder:
    """BLAKE2b signed feature hashing with stable cross-process results."""

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions < 16:
            raise ValueError("dimensions must be >= 16")
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        """Tokenize words/individual Han characters, hash dimensions, and L2-normalize."""
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[\w]+|[\u4e00-\u9fff]", text.lower(), flags=re.UNICODE)
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            # An independent sign bit prevents collisions from adding only positive bias.
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute cosine similarity; empty vectors return zero and dimensions must match."""
    if len(left) != len(right):
        raise ValueError("embedding dimensions differ")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    raw = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    return max(-1.0, min(1.0, raw))


def bounded_similarity(left: str, right: str, embedder: Embedder) -> float:
    """Map cosine values linearly from ``[-1, 1]`` to the ``[0, 1]`` feature range."""
    return (cosine_similarity(embedder.embed(left), embedder.embed(right)) + 1.0) / 2.0
