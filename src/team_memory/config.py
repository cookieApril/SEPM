"""集中定义运行时策略和可复现实验阈值。

配置对象不可变（``frozen=True``），避免服务运行期间某个 worker 偷偷改变门控规则。
生产环境可显式构造 ``MemoryConfig`` 覆盖默认值；默认值服务于本地研究原型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class MemoryConfig:
    """服务策略快照；所有可消融阈值都集中在此，便于记录实验条件。"""

    # 存储与三类偏差检测。
    database_path: Path = Path("team_memory.db")
    goal_embedding_weight: float = 0.55
    goal_divergence_threshold: float = 0.35
    plan_divergence_threshold: float = 0.40
    state_confidence_threshold: float = 0.70
    # SOP 验证/晋升门。默认首个合格 trial 即可发布。
    min_reproductions: int = 1
    min_distinct_task_families: int = 1
    min_causal_confidence: float = 0.65
    min_validation_score: float = 0.75
    min_net_benefit: float = 0.0
    utility_cost_weight: float = 1.0
    safety_risk_threshold: float = 0.50
    # 存储容量保护。
    max_blackboard_entries: int = 10_000
    # 以下开关只用于受控消融。默认值全部保持完整方法；论文 runner 会把所选 variant
    # 映射为新的不可变 MemoryConfig，绝不在运行中修改同一个 service 的实验条件。
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
    # 这些词只负责把步骤标记为敏感；是否拒绝仍由 SafetyValidator 的规则决定。
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
