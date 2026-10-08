"""Resolve shared world state by evidence authority.

The order is Authoritative State > Tool Observation > Verified Artifact > Agent
Inference. Equal tiers are ordered by time. Conflicts at the same tier and time
require a new query; agent voting never guesses external facts.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from .config import MemoryConfig
from .models import ConflictType, Evidence, EvidenceTier, Resolution


def _canonical(value: Any) -> str:
    """Serialize JSON-like values stably so key order creates no false conflict."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


class EvidenceResolver:
    """Filter by authority and confidence, remaining unresolved without a unique result."""

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self.config = config or MemoryConfig()

    def resolve_world_state(self, state_key: str, evidence: list[Evidence]) -> Resolution:
        """Resolve one state key and return the winner, evidence, and next action."""
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
            # Ablation baseline: ignore authority and vote by normalized value. Ties
            # still abstain so random tie-breaking does not introduce seed noise.
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
        # Only the highest authority tier competes; lower tiers cannot win by volume.
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
            # Keep the newest observation for each conflicting value, then compare times.
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
