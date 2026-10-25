"""Hybrid BM25/VectorSearch retrieval with query-adaptive weights (Eq. 3).

Retrieval answers which SOP is most likely to help the current task by combining
lexical BM25 relevance with vector semantic relevance. Only active SOPs whose
context conditions are verified facts and whose exclusions do not apply compete.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from .config import MemoryConfig
from .embedding import Embedder, bounded_similarity
from .models import RetrievalResult, SOPVersion


class AdaptiveWeightRouter:
    """Softmax router over BM25 and vector logits driven by query cues (Appendix B.1, Table 3)."""

    LABELS = ("bm25", "vector")
    # (cues, modified logit, increment); Chinese cues mirror the English ones.
    ROUTING_CUES: tuple[tuple[tuple[str, ...], str, float], ...] = (
        (("code", "debug", "implement", "coding", "代码", "调试"), "vector", 1.0),
        (("latest", "current", "real-time", "实时", "最新"), "bm25", 0.8),
        (("routine", "repeat", "standard", "日常", "重复", "标准"), "bm25", 0.6),
    )

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self.config = config or MemoryConfig()

    def weights(self, query: str) -> dict[str, float]:
        """Compute w_b and w_v; both are 0.50 without a cue or with adaptive retrieval disabled."""
        if not self.config.adaptive_retrieval:
            return {"bm25": 0.50, "vector": 0.50}
        lower = query.lower()
        logits = {label: 0.0 for label in self.LABELS}
        for cues, label, increment in self.ROUTING_CUES:
            if any(cue in lower for cue in cues):
                logits[label] += increment
        probabilities = self._softmax([logits[label] for label in self.LABELS])
        return {label: value for label, value in zip(self.LABELS, probabilities)}

    @staticmethod
    def _softmax(values: Sequence[float]) -> list[float]:
        """Subtract the largest logit before exponentiation to avoid overflow."""
        largest = max(values)
        exps = [math.exp(value - largest) for value in values]
        total = sum(exps)
        return [value / total for value in exps]


class SOPRetriever:
    """Filter compatible SOPs, apply query-dependent weights, and return the top K."""
    def __init__(
        self,
        embedder: Embedder,
        router: AdaptiveWeightRouter | None = None,
        config: MemoryConfig | None = None,
    ) -> None:
        self.embedder = embedder
        self.config = config or MemoryConfig()
        self.router = router or AdaptiveWeightRouter(self.config)

    def rank(
        self,
        query: str,
        sops: list[SOPVersion],
        limit: int = 5,
        *, context_facts: set[str] | None = None,
    ) -> list[RetrievalResult]:
        """Rank the candidate set C(q, F_q) by s_i(q) and return TopK."""
        weights = self.router.weights(query)
        facts = {fact.strip().casefold() for fact in (context_facts or set())}
        # Conditions are verified context facts, not positive lexical ranking features.
        sops = [sop for sop in sops if sop.active
                and {c.casefold() for c in sop.metadata.context_conditions}.issubset(facts)
                and not ({c.casefold() for c in sop.metadata.exclusions} & facts)]
        documents = [self._document(sop) for sop in sops]
        bm25_scores = self._bm25_scores(query, documents)
        results: list[RetrievalResult] = []
        for index, sop in enumerate(sops):
            features = {
                "bm25": bm25_scores[index],
                "vector": bounded_similarity(query, documents[index], self.embedder),
            }
            score = sum(weights[name] * features[name] for name in weights)
            results.append(
                RetrievalResult(
                    sop=sop,
                    score=max(0.0, min(1.0, score)),
                    features=features,
                    weights=weights,
                    explanation=(
                        "s(q)=w_b*BM25(q)+w_v*sim(f(q),f(S)) over SOPs whose context conditions "
                        "hold and whose exclusions do not apply"
                    ),
                )
            )
        return sorted(results, key=lambda result: result.score, reverse=True)[:limit]

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[\w一-鿿]+", text.lower())

    @staticmethod
    def _document(sop: SOPVersion) -> str:
        return " ".join(
            [
                sop.metadata.title,
                sop.metadata.description,
                sop.metadata.task_family,
                *sop.metadata.applicability,
                *sop.metadata.tags,
                *sop.metadata.context_conditions,
                *(warning.text for warning in sop.metadata.warnings),
                *(step.instruction for step in sop.procedure.steps),
            ]
        )

    def _bm25_scores(self, query: str, documents: list[str]) -> list[float]:
        """BM25 with k1=1.5, b=0.75, normalized by the maximum in the candidate set."""
        query_terms = self._tokens(query)
        tokenized = [self._tokens(document) for document in documents]
        if not query_terms or not tokenized:
            return [0.0 for _ in documents]
        doc_freq = Counter(term for terms in tokenized for term in set(terms))
        avg_len = sum(len(terms) for terms in tokenized) / max(1, len(tokenized))
        k1 = 1.5
        b = 0.75
        raw_scores: list[float] = []
        for terms in tokenized:
            counts = Counter(terms)
            doc_len = len(terms)
            score = 0.0
            for term in query_terms:
                if counts[term] == 0:
                    continue
                idf = math.log(1.0 + (len(documents) - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
                denom = counts[term] + k1 * (1.0 - b + b * doc_len / max(1e-12, avg_len))
                score += idf * (counts[term] * (k1 + 1.0)) / denom
            raw_scores.append(score)
        max_score = max(raw_scores, default=0.0)
        return [score / max_score if max_score else 0.0 for score in raw_scores]
