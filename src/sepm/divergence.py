"""Detect divergence from an agent's assigned goal, assigned plan, and shared state.

Goal divergence (Eq. 5) combines embedding distance with a replaceable
goal-consistency judge. Plan divergence (Eq. 6) compares the agent's local plan
with its assigned subgraph under fixed node correspondence, and the service
supplies state conflicts (Eq. 7). This module reports divergence but never
mutates canonical state.
"""

from __future__ import annotations

import re
from typing import ClassVar, Protocol

from .config import MemoryConfig
from .embedding import Embedder, bounded_similarity
from .models import BlackboardEntry, DivergenceReport, TaskPlan, Workspace


class GoalJudge(Protocol):
    """Goal-consistency protocol J_goal, replaceable by a frozen LLM judge in experiments."""
    def consistency_score(self, agent_goal: str, assigned_goal: str, task_objective: str) -> float:
        """Return consistency in [0, 1] of the agent goal with its assignment and the task objective."""


class LexicalGoalJudge:
    """Offline lexical fallback that checks negation before token Jaccard similarity.

    It compares the agent goal with the assigned goal only; an LLM judge can also
    use ``task_objective`` to check global constraints.
    """

    NEGATIONS: ClassVar[set[str]] = {"not", "never", "without"}

    def consistency_score(self, agent_goal: str, assigned_goal: str, task_objective: str) -> float:
        """Return consistency in ``[0, 1]``; different negation sets are inconsistent."""
        assigned = assigned_goal.lower()
        agent = agent_goal.lower()
        def negations(text: str) -> set[str]:
            return {word for word in self.NEGATIONS if
                    (re.search(r"\b" + re.escape(word) + r"\b", text) if word.isascii() else word in text)}
        if negations(assigned) != negations(agent):
            return 0.0
        assigned_terms = set(assigned.split())
        agent_terms = set(agent.split())
        if not assigned_terms or not agent_terms:
            return 0.0
        return len(assigned_terms & agent_terms) / len(assigned_terms | agent_terms)


def normalized_plan_distance(left: TaskPlan, right: TaskPlan) -> float:
    """Compute D_p (Eq. 6) under fixed node correspondence by persistent node ID.

    A different action at a matched ID counts as one node edit, and a node present
    in only one plan is compared against a null action. Each edge in the symmetric
    difference counts as one edit. The sum is normalized by ``|V| + |E|`` of the union.
    """
    left_actions = {node.node_id: node.action.strip().lower() for node in left.nodes}
    right_actions = {node.node_id: node.action.strip().lower() for node in right.nodes}
    node_ids = set(left_actions) | set(right_actions)
    node_edits = sum(left_actions.get(node_id) != right_actions.get(node_id) for node_id in node_ids)
    left_edges = {(edge.source, edge.target) for edge in left.edges}
    right_edges = {(edge.source, edge.target) for edge in right.edges}
    edge_edits = len(left_edges ^ right_edges)
    denominator = max(1, len(node_ids) + len(left_edges | right_edges))
    return (node_edits + edge_edits) / denominator


class DivergenceDetector:
    """Combine three divergence signals and decide whether alignment is required (Eq. 7)."""
    def __init__(
        self,
        embedder: Embedder,
        goal_judge: GoalJudge | None = None,
        config: MemoryConfig | None = None,
    ) -> None:
        self.embedder = embedder
        self.goal_judge = goal_judge or LexicalGoalJudge()
        self.config = config or MemoryConfig()

    def inspect(
        self,
        workspace: Workspace,
        agent_entry: BlackboardEntry,
        state_conflicts: list[str],
        reference_plan: TaskPlan | None = None,
    ) -> DivergenceReport:
        """Compare the agent's latest goal/plan with its assignment in the workspace.

        ``reference_plan`` overrides the assigned subgraph, e.g. with the part of the
        task graph an agent has reached so far (see ``alignment.derive_alignment``).
        """
        # A local subtask is not a declaration that the agent's goal changed.
        goal_observed = agent_entry.goal is not None
        goal_divergence = embedding_divergence = judge_divergence = 0.0
        if goal_observed:
            assigned_goal = workspace.reference_goal(agent_entry.agent_id)
            embedding_divergence = 1.0 - bounded_similarity(agent_entry.goal, assigned_goal, self.embedder)
            judge_divergence = 1.0 - self.goal_judge.consistency_score(
                agent_entry.goal, assigned_goal, workspace.main_goal
            )
            weight = self.config.goal_embedding_weight
            goal_divergence = weight * embedding_divergence + (1.0 - weight) * judge_divergence
        if reference_plan is None:
            reference_plan = workspace.reference_plan(agent_entry.agent_id)
        plan_divergence = (
            normalized_plan_distance(reference_plan, agent_entry.plan)
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
            action = "realign agent goal or task dependency graph against its assignment"
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
