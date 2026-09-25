"""提供无外部模型依赖、跨进程确定性的文本向量实现。

HashingEmbedder 用于本地测试和论文基线：它速度快、无需下载模型，但不应被误解为
生产级语义 embedding。``Embedder`` 协议允许调用方注入真实向量模型。
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol


class Embedder(Protocol):
    """所有 embedding 后端必须实现的最小接口。"""
    def embed(self, text: str) -> list[float]: ...


class HashingEmbedder:
    """使用 BLAKE2b 的 signed feature hashing，保证跨进程结果一致。"""

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions < 16:
            raise ValueError("dimensions must be >= 16")
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        """按英文单词/单个汉字分词、哈希到固定维度，并做 L2 归一化。"""
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[\w]+|[\u4e00-\u9fff]", text.lower(), flags=re.UNICODE)
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            # 独立符号位降低哈希碰撞只会正向累加造成的系统性偏差。
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """计算余弦相似度；空向量返回 0，维度不一致则拒绝比较。"""
    if len(left) != len(right):
        raise ValueError("embedding dimensions differ")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    raw = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    return max(-1.0, min(1.0, raw))


def bounded_similarity(left: str, right: str, embedder: Embedder) -> float:
    """把余弦值从 ``[-1, 1]`` 线性映射到统一的 ``[0, 1]`` 特征区间。"""
    return (cosine_similarity(embedder.embed(left), embedder.embed(right)) + 1.0) / 2.0
