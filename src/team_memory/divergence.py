"""检测 Agent 与规范目标、计划图和共享世界状态之间的偏差。

目标偏差混合 embedding 距离和可替换 judge；计划偏差使用轻量 graph-edit
proxy；状态冲突由服务层整理后传入。该模块只生成报告，不会擅自修改规范状态。
"""

from __future__ import annotations

from typing import Protocol
from collections import Counter
import re

from .config import MemoryConfig
from .embedding import Embedder, bounded_similarity
from .models import BlackboardEntry, DivergenceReport, TaskPlan, Workspace


class GoalJudge(Protocol):
    """目标一致性判断器协议，实验时可替换为冻结的 LLM judge。"""
    def consistency_score(self, canonical_goal: str, agent_goal: str) -> float:
        """Return consistency in [0, 1], where 1 means fully consistent."""


class LexicalGoalJudge:
    """离线词法后备实现；先检查否定语义，再计算 token Jaccard 相似度。"""

    NEGATIONS = {"not", "never", "without", "禁止", "不要", "不得", "无需"}

    def consistency_score(self, canonical_goal: str, agent_goal: str) -> float:
        """返回 ``[0, 1]`` 一致性；否定词集合不同会直接判为不一致。"""
        canonical = canonical_goal.lower()
        agent = agent_goal.lower()
        def negations(text: str) -> set[str]:
            return {word for word in self.NEGATIONS if
                    (re.search(r"\b" + re.escape(word) + r"\b", text) if word.isascii() else word in text)}
        canonical_neg = negations(canonical)
        agent_neg = negations(agent)
        if canonical_neg != agent_neg:
            return 0.0
        canonical_terms = set(canonical.split())
        agent_terms = set(agent.split())
        if not canonical_terms or not agent_terms:
            return 0.0
        return len(canonical_terms & agent_terms) / len(canonical_terms | agent_terms)


def normalized_plan_distance(left: TaskPlan, right: TaskPlan) -> float:
    """计算动作标注节点和依赖边上的归一化 graph-edit 近似距离。

    同 id 节点的动作变化计一次编辑，边集合的对称差各计一次；结果裁剪到
    ``[0, 1]``。这是快速、确定性的 proxy，并非昂贵的精确 GED。
    """
    left_actions = {node.node_id: node.action.strip().lower() for node in left.nodes}
    right_actions = {node.node_id: node.action.strip().lower() for node in right.nodes}
    # Match uniquely labeled actions before falling back to stable node identifiers.
    # This removes spurious edits when the same plan is serialized with fresh IDs.
    left_counts, right_counts = Counter(left_actions.values()), Counter(right_actions.values())
    by_action = {action: node for node, action in left_actions.items() if left_counts[action] == 1}
    mapping = {node: by_action[action] for node, action in right_actions.items()
               if action in by_action and right_counts[action] == 1}
    used = set(mapping.values())
    for node in right_actions:
        if node not in mapping:
            mapping[node] = node if node in left_actions and node not in used else ("right", node)
            used.add(mapping[node])
    right_actions = {mapping[node]: action for node, action in right_actions.items()}
    node_ids = set(left_actions) | set(right_actions)
    node_edits = sum(left_actions.get(node_id) != right_actions.get(node_id) for node_id in node_ids)
    left_edges = {(edge.source, edge.target) for edge in left.edges}
    right_edges = {(mapping[edge.source], mapping[edge.target]) for edge in right.edges}
    edge_edits = len(left_edges ^ right_edges)
    denominator = max(1, len(node_ids) + len(left_edges | right_edges))
    return min(1.0, (node_edits + edge_edits) / denominator)


class DivergenceDetector:
    """组合三个偏差信号，并根据配置阈值决定是否需要 Agent 对齐。"""
    def __init__(
        self,
        embedder: Embedder,
        goal_judge: GoalJudge | None = None,
        config: MemoryConfig | None = None,
    ) -> None:
        self.embedder = embedder
        self.goal_judge = goal_judge or LexicalGoalJudge()
        self.config = config or MemoryConfig()

    def inspect(self, workspace: Workspace, agent_entry: BlackboardEntry, state_conflicts: list[str]) -> DivergenceReport:
        """比较 Agent 最新条目与工作区规范状态，返回结构化告警原因。"""
        # A local subtask is not a declaration that the global objective changed.
        goal_observed = agent_entry.goal is not None
        agent_goal = agent_entry.goal or ""
        similarity = bounded_similarity(workspace.main_goal, agent_goal, self.embedder)
        embedding_divergence = 1.0 - similarity
        judge_consistency = self.goal_judge.consistency_score(workspace.main_goal, agent_goal)
        judge_divergence = 1.0 - judge_consistency
        weight = self.config.goal_embedding_weight
        # 两种信号互补：embedding 覆盖语义近似，judge 负责显式逻辑/否定差异。
        goal_divergence = weight * embedding_divergence + (1.0 - weight) * judge_divergence
        if not goal_observed:
            goal_divergence = embedding_divergence = judge_divergence = 0.0
        plan_divergence = (
            normalized_plan_distance(workspace.plan, agent_entry.plan)
            if agent_entry.plan is not None
            else 0.0
        )
        reasons: list[str] = []
        if (
            self.config.detect_goal_divergence
            and goal_observed
            and goal_divergence > self.config.goal_divergence_threshold
        ):
            reasons.append("goal_divergence")
        if (
            self.config.detect_plan_divergence
            and plan_divergence > self.config.plan_divergence_threshold
        ):
            reasons.append("plan_divergence")
        visible_state_conflicts = (
            state_conflicts if self.config.detect_world_state_divergence else []
        )
        if visible_state_conflicts:
            reasons.append("world_state_conflict")
        if not reasons:
            stage = "none"
            action = "no realignment required"
        elif visible_state_conflicts:
            stage = "verify"
            action = "verify conflicting state with authoritative/tool evidence before negotiation"
        else:
            stage = "realign"
            action = "realign agent goal or task dependency graph against the workspace plan"
        return DivergenceReport(
            agent_id=agent_entry.agent_id,
            goal_observed=goal_observed,
            plan_observed=agent_entry.plan is not None,
            goal_divergence=goal_divergence,
            goal_embedding_component=embedding_divergence,
            goal_judge_component=judge_divergence,
            plan_divergence=plan_divergence,
            conflicting_state_keys=visible_state_conflicts,
            requires_alignment=bool(reasons),
            reasons=reasons,
            alignment_stage=stage,
            recommended_action=action,
        )
