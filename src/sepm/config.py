"""Central definitions for runtime policy and reproducible experiment thresholds.

The configuration is immutable (``frozen=True``), preventing workers from
changing gates during a run. Production callers may construct ``MemoryConfig``
with explicit overrides; defaults target the local research prototype.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class MemoryConfig:
    """Service policy snapshot containing every ablatable experiment threshold."""

    # Storage and the three divergence detectors.
    database_path: Path = Path("sepm.db")
    goal_embedding_weight: float = 0.55
    goal_divergence_threshold: float = 0.35
    plan_divergence_threshold: float = 0.40
    state_confidence_threshold: float = 0.70
    # SOP validation and promotion gates. The first qualified trial publishes by default.
    min_reproductions: int = 1
    min_distinct_task_families: int = 1
    min_causal_confidence: float = 0.65
    min_validation_score: float = 0.75
    min_net_benefit: float = 0.0
    utility_cost_weight: float = 1.0
    safety_risk_threshold: float = 0.50
    # Storage capacity guard.
    max_blackboard_entries: int = 10_000
    # These switches are for controlled ablations only. Defaults preserve the full method;
    # the paper runner maps a variant to a new immutable configuration without mutating a run.
    enable_procedural_memory: bool = True
    share_blackboard_across_agents: bool = True
    use_evidence_hierarchy: bool = True
    enable_divergence_detection: bool = True
    detect_goal_divergence: bool = True
    detect_plan_divergence: bool = True
    detect_world_state_divergence: bool = True
    require_causal_gate: bool = True
    enable_safety_gate: bool = True
    adaptive_retrieval: bool = True
    # These terms only mark sensitive steps; SafetyValidator rules decide rejection.
    safety_keywords: tuple[str, ...] = field(
        default=(
            "payment",
            "pay",
            "delete",
            "remove",
            "drop table",
            "migration",
            "credential",
            "permission",
            "allergy",
            "medical",
            "transfer",
            "authorize",
        )
    )
