"""对已发布 SOP 进行 BM25 + VectorSearch 混合召回和轻量在线权重学习。

检索只回答“当前任务最可能要用哪个 SOP”：默认综合词法 BM25 相关性和向量语义相关性。
安全关键步骤不参与检索加权；它们由 SOP 更新、删除和 supersede 时的安全门保护。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from .config import MemoryConfig
from .embedding import Embedder, bounded_similarity
from .models import RetrievalResult, SOPVersion, TaskPlan


class AdaptiveWeightRouter:
    """A tiny trainable softmax router over BM25 and vector relevance.

    `learn` provides online SGD for an experimental learned router; deterministic
    task priors make the untrained model useful.
    """

    LABELS = ("bm25", "vector")

    def __init__(self, embedder: Embedder, config: MemoryConfig | None = None) -> None:
        self.embedder = embedder
        self.config = config or MemoryConfig()
        self.bias = {label: 0.0 for label in self.LABELS}
        self.hidden_size = 8
        self.output_weights = {label: [0.0] * self.hidden_size for label in self.LABELS}

    def weights(self, query: str) -> dict[str, float]:
        """根据查询计算归一化权重，返回值总和恒为 1。"""
        if not self.config.adaptive_retrieval:
            return {"bm25": 0.50, "vector": 0.50}
        lower = query.lower()
        hidden = self._hidden(query)
        logits = {
            label: self.bias[label]
            + sum(weight * activation for weight, activation in zip(self.output_weights[label], hidden))
            for label in self.LABELS
        }
        # 这些先验让未经训练的 router 也有合理行为；在线学习在其上继续调整。
        if any(word in lower for word in ("code", "debug", "implement", "coding", "代码", "调试")):
            logits["vector"] += 1.0
        if any(word in lower for word in ("latest", "current", "real-time", "today", "实时", "最新")):
            logits["bm25"] += 0.8
        if any(word in lower for word in ("routine", "repeat", "standard", "日常", "重复", "标准")):
            logits["bm25"] += 0.6
        probabilities = self._softmax([logits[label] for label in self.LABELS])
        return {label: value for label, value in zip(self.LABELS, probabilities)}

    def learn(self, query: str, preferred_feature: str, learning_rate: float = 0.05) -> None:
        """用交叉熵梯度向用户偏好的检索特征做一步在线更新。"""
        if not self.config.adaptive_retrieval:
            return
        if preferred_feature not in self.LABELS:
            raise ValueError(f"preferred_feature must be one of {self.LABELS}")
        hidden = self._hidden(query)
        weights = self.weights(query)
        for label in self.LABELS:
            probability = weights[label]
            target = 1.0 if label == preferred_feature else 0.0
            gradient = target - probability
            self.bias[label] += learning_rate * gradient
            self.output_weights[label] = [
                weight + learning_rate * gradient * activation
                for weight, activation in zip(self.output_weights[label], hidden)
            ]

    def _hidden(self, query: str) -> list[float]:
        """Fixed signed projection + tanh hidden layer over the task embedding."""
        embedding = self.embedder.embed(query)
        scale = math.sqrt(max(1, len(embedding)))
        return [
            math.tanh(
                sum(
                    value * math.sin((index + 1) * (unit + 1) * 0.017) / scale
                    for index, value in enumerate(embedding)
                )
            )
            for unit in range(self.hidden_size)
        ]

    @staticmethod
    def _softmax(values: Sequence[float]) -> list[float]:
        """减去最大 logit 后再指数化，避免数值溢出。"""
        largest = max(values)
        exps = [math.exp(value - largest) for value in values]
        total = sum(exps)
        return [value / total for value in exps]


class SOPRetriever:
    """把每个 SOP 转为特征向量，应用动态权重并返回最高分结果。"""
    def __init__(
        self,
        embedder: Embedder,
        router: AdaptiveWeightRouter | None = None,
        config: MemoryConfig | None = None,
    ) -> None:
        self.embedder = embedder
        self.config = config or MemoryConfig()
        self.router = router or AdaptiveWeightRouter(embedder, self.config)

    def rank(
        self,
        query: str,
        sops: list[SOPVersion],
        limit: int = 5,
        query_plan: TaskPlan | None = None,
        *, context_facts: set[str] | None = None,
    ) -> list[RetrievalResult]:
        """对活动 SOP 排序；query_plan 为兼容旧调用保留，但不参与检索打分。"""
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
            document = documents[index]
            vector_relevance = bounded_similarity(query, document, self.embedder)
            bm25_relevance = bm25_scores[index]
            features = {
                "bm25": bm25_relevance,
                "vector": vector_relevance,
                "bm25_relevance": bm25_relevance,
                "vector_relevance": vector_relevance,
            }
            score = sum(weights[name] * features[name] for name in weights)
            results.append(
                RetrievalResult(
                    sop=sop,
                    score=max(0.0, min(1.0, score)),
                    features=features,
                    weights=weights,
                    explanation=(
                        "Candidate(q)=BM25(q) union VectorSearch(q); ranking combines lexical and semantic relevance; "
                        "safety-critical steps are protected during SOP updates/deletes"
                    ),
                )
            )
        return sorted(results, key=lambda result: result.score, reverse=True)[:limit]

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[\w\u4e00-\u9fff]+", text.lower())

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
                *sop.metadata.warnings,
                *(step.instruction for step in sop.procedure.steps),
            ]
        )

    def _bm25_scores(self, query: str, documents: list[str]) -> list[float]:
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
