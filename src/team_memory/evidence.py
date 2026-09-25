"""按证据权威等级裁决共享世界状态。

裁决顺序是 Authoritative State > Tool Observation > Verified Artifact > Agent
Inference。相同等级先比较时间；同等级同时间仍互相冲突时，明确要求重新查询，绝不
通过 Agent 投票猜测外部事实。
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from .config import MemoryConfig
from .models import ConflictType, Evidence, EvidenceTier, Resolution


def _canonical(value: Any) -> str:
    """把任意 JSON-like 值稳定序列化，使字典键顺序不制造伪冲突。"""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


class EvidenceResolver:
    """先按权威等级和置信度筛选证据，无法唯一裁决时保持 unresolved。"""

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self.config = config or MemoryConfig()

    def resolve_world_state(self, state_key: str, evidence: list[Evidence]) -> Resolution:
        """裁决一个 state key，并返回赢家、全部证据及建议的下一步动作。"""
        if not evidence:
            return Resolution(
                conflict_type=ConflictType.WORLD_STATE,
                state_key=state_key,
                resolved=False,
                considered_evidence=[],
                next_action="query an authoritative state source or verified tool",
            )
        qualified = [item for item in evidence if item.confidence >= self.config.state_confidence_threshold]
        if not qualified:
            return Resolution(
                conflict_type=ConflictType.WORLD_STATE,
                state_key=state_key,
                resolved=False,
                considered_evidence=evidence,
                next_action="collect higher-confidence evidence",
            )
        if not self.config.use_evidence_hierarchy:
            # 消融基线：忽略来源权威等级，按规范化值做简单多数投票。平票时仍弃权，
            # 避免随机 tie-breaking 把 seed 噪声引入组件贡献。
            groups: dict[str, list[Evidence]] = defaultdict(list)
            for item in qualified:
                groups[_canonical(item.value)].append(item)
            ordered = sorted(groups.values(), key=lambda items: len(items), reverse=True)
            if len(ordered) > 1 and len(ordered[0]) == len(ordered[1]):
                return Resolution(
                    conflict_type=ConflictType.WORLD_STATE,
                    state_key=state_key,
                    resolved=False,
                    considered_evidence=evidence,
                    next_action="collect another observation; majority vote is tied",
                )
            winner = max(ordered[0], key=lambda item: (item.confidence, item.observed_at))
            return Resolution(
                conflict_type=ConflictType.WORLD_STATE,
                state_key=state_key,
                resolved=True,
                value=winner.value,
                winning_evidence=winner,
                considered_evidence=evidence,
                next_action="write the majority value to shared task state",
            )
        # 只允许最高权威等级参与最终比较，低等级证据不能靠数量压过高等级证据。
        top_tier = max(item.tier for item in qualified)
        if top_tier == EvidenceTier.AGENT_INFERENCE:
            return Resolution(
                conflict_type=ConflictType.WORLD_STATE, state_key=state_key,
                resolved=False, considered_evidence=evidence,
                next_action="query external state if available; otherwise negotiate with explicit evidence",
            )
        top = [item for item in qualified if item.tier == top_tier]
        groups: dict[str, list[Evidence]] = defaultdict(list)
        for item in top:
            groups[_canonical(item.value)].append(item)
        if len(groups) > 1:
            if len({item.source for item in top}) > 1:
                return Resolution(
                    conflict_type=ConflictType.WORLD_STATE, state_key=state_key,
                    resolved=False, considered_evidence=evidence,
                    next_action="re-query the authoritative source; equal-tier sources conflict",
                )
            # 每个互斥值只保留最新观测，再在这些值之间比较时间。
            newest_by_value = {
                value: max(items, key=lambda item: item.observed_at) for value, items in groups.items()
            }
            ordered = sorted(newest_by_value.values(), key=lambda item: item.observed_at, reverse=True)
            if len(ordered) > 1 and ordered[0].observed_at == ordered[1].observed_at:
                return Resolution(
                    conflict_type=ConflictType.WORLD_STATE,
                    state_key=state_key,
                    resolved=False,
                    considered_evidence=evidence,
                    next_action="re-query the authoritative source; equal-tier evidence conflicts",
                )
            winner = ordered[0]
        else:
            winner = max(top, key=lambda item: (item.confidence, item.observed_at))
        return Resolution(
            conflict_type=ConflictType.WORLD_STATE,
            state_key=state_key,
            resolved=True,
            value=winner.value,
            winning_evidence=winner,
            considered_evidence=evidence,
            next_action="write the verified value to shared task state",
        )
