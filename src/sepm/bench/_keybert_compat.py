"""KeyBERT stand-in for MARBLE's research tools on machines without torch.

Implements KeyBERT's default ``extract_keywords`` (CountVectorizer n-gram
candidates ranked by cosine similarity to the document embedding, no MMR/MaxSum)
with an injected embedder instead of a local sentence-transformer.
"""

from __future__ import annotations

import sys
import types
from typing import Any

_EMBEDDER: Any = None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class KeyBERT:
    def __init__(self, model: Any = None) -> None:
        if _EMBEDDER is None:
            raise RuntimeError("KeyBERT stand-in requires install_keybert(embedder) first")
        self.embedder = _EMBEDDER

    def extract_keywords(
        self,
        docs: str,
        keyphrase_ngram_range: tuple[int, int] = (1, 1),
        stop_words: str | None = "english",
        top_n: int = 5,
        **_: Any,
    ) -> list[tuple[str, float]]:
        from sklearn.feature_extraction.text import CountVectorizer

        try:
            vectorizer = CountVectorizer(ngram_range=keyphrase_ngram_range, stop_words=stop_words).fit([docs])
        except ValueError:
            return []
        candidates = list(vectorizer.get_feature_names_out())[:300]
        if not candidates:
            return []
        doc_vector = self.embedder.embed(docs[:8000])
        vectors = self.embedder.embed_documents(candidates)
        scored = sorted(
            ((word, round(_cosine(doc_vector, vector), 4)) for word, vector in zip(candidates, vectors)),
            key=lambda item: item[1],
            reverse=True,
        )
        return scored[:top_n]


def install_keybert(embedder: Any) -> None:
    global _EMBEDDER
    _EMBEDDER = embedder
    module = types.ModuleType("keybert")
    module.KeyBERT = KeyBERT
    sys.modules["keybert"] = module
