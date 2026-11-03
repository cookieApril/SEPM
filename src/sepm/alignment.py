"""Derive one agent's local plan from its blackboard actions (Section 4.2).

The reference is the task graph instantiated from a retrieved procedure. Each
executed action is matched to a graph node; neutral actions such as navigation
or looking around are ignored. The local plan holds the nodes the agent has
attempted, the prerequisite edges verified before its latest attempt of each
node, and state-changing actions outside the procedure since its last verified
progress. The reference view is the part of the graph the agent has reached:
attempted nodes and all their prerequisites. Comparing the two with Eq. 6
measures skipped prerequisites and off-plan actions without penalizing steps
the agent has not reached yet.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from .embedding import cosine_similarity
from .models import PlanEdge, PlanNode, TaskPlan

STOPWORDS = frozenset(
    {"a", "an", "the", "to", "in", "on", "of", "from", "with", "and", "then", "it", "them",
     "some", "any", "into", "onto", "at", "for", "by", "your", "you", "is", "are", "be"}
)


@dataclass(frozen=True)
class ActionRecord:
    """One executed action and the observation it produced."""
    action: str
    observation: str = ""


class ActionMatcher(Protocol):
    """Domain rules for reading actions against procedure steps."""
    def is_neutral(self, action: str) -> bool:
        """True for actions that gather information without changing task state."""
    def failed(self, observation: str) -> bool:
        """True when the observation shows the action had no effect."""
    def match(self, action: str, plan: TaskPlan) -> str | None:
        """Return the node an action executes, or None when it is off-plan."""


@dataclass(frozen=True)
class LexicalActionMatcher:
    """Match actions to steps by shared verb and content-word overlap."""
    neutral_verbs: frozenset[str] = frozenset({"look", "inventory", "think"})
    failure_markers: tuple[str, ...] = ()
    verb_aliases: Mapping[str, str] = field(default_factory=dict)
    min_overlap: float = 0.5

    def tokens(self, text: str) -> list[str]:
        words = re.findall(r"[a-z]+", text.lower())
        return [self.verb_aliases.get(word, word) for word in words if word not in STOPWORDS]

    def is_neutral(self, action: str) -> bool:
        words = self.tokens(action)
        return not words or words[0] in self.neutral_verbs

    def failed(self, observation: str) -> bool:
        lower = observation.lower()
        return any(marker in lower for marker in self.failure_markers)

    def match(self, action: str, plan: TaskPlan) -> str | None:
        words = self.tokens(action)
        if not words:
            return None
        verb, content = words[0], set(words)
        best, best_score = None, 0.0
        for node in plan.nodes:
            node_words = set(self.tokens(node.action))
            if verb not in node_words:
                continue
            score = len(content & node_words) / len(content)
            if score >= self.min_overlap and score > best_score:
                best, best_score = node.node_id, score
        return best


@dataclass(frozen=True)
class EmbeddingActionMatcher:
    """Match free-text actions (messages, tool calls) to steps by embedding similarity.

    Text hosts have no leading command verb, so lexical verb matching rarely fires;
    an action executes the most similar step when cosine similarity reaches ``threshold``.
    """
    embedder: Any
    failure_markers: tuple[str, ...] = ()
    threshold: float = 0.5
    # A free-text message that executes no step is conversation, not an off-plan action.
    unmatched_is_neutral: bool = True

    def is_neutral(self, action: str) -> bool:
        return not action.strip()

    def failed(self, observation: str) -> bool:
        lower = observation.lower()
        return any(marker in lower for marker in self.failure_markers)

    def match(self, action: str, plan: TaskPlan) -> str | None:
        if not action.strip() or not plan.nodes:
            return None
        vector = self.embedder.embed(action[:2_000])
        best, best_score = None, self.threshold
        for node in plan.nodes:
            score = cosine_similarity(vector, self.embedder.embed(node.action[:2_000]))
            if score >= best_score:
                best, best_score = node.node_id, score
        return best


@dataclass(frozen=True)
class AlignmentView:
    """Reference view and local plan for Eq. 6, with the edits that separate them."""
    reference_view: TaskPlan
    local_plan: TaskPlan
    missing_prerequisites: list[tuple[str, str]]
    off_plan_actions: list[str]
    verified_nodes: frozenset[str]
    step_count: int


def contract_neutral_steps(plan: TaskPlan, matcher: ActionMatcher) -> TaskPlan:
    """Remove steps no state-changing action can execute, keeping their prerequisite chains."""
    nodes = [node for node in plan.nodes if not matcher.is_neutral(node.action)]
    kept = {node.node_id for node in nodes}
    successors: dict[str, set[str]] = {node.node_id: set() for node in plan.nodes}
    for edge in plan.edges:
        successors[edge.source].add(edge.target)

    def reachable_kept(start: str) -> set[str]:
        found: set[str] = set()
        stack = list(successors[start])
        seen: set[str] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            if current in kept:
                found.add(current)
            else:
                stack.extend(successors[current])
        return found

    edges = sorted({(source, target) for source in kept for target in reachable_kept(source)})
    return TaskPlan(nodes=nodes, edges=[PlanEdge(source=s, target=t) for s, t in edges])


def derive_alignment(reference: TaskPlan, records: list[ActionRecord], matcher: ActionMatcher) -> AlignmentView:
    """Build the reference view and the agent's local plan from its executed actions."""
    graph = contract_neutral_steps(reference, matcher)
    actions = {node.node_id: node.action for node in graph.nodes}
    predecessors: dict[str, list[str]] = {node_id: [] for node_id in actions}
    for edge in graph.edges:
        predecessors[edge.target].append(edge.source)
    verified_at: dict[str, int] = {}
    last_attempt: dict[str, int] = {}
    off_plan: list[tuple[int, str]] = []
    last_progress = -1
    for index, record in enumerate(records):
        if matcher.is_neutral(record.action):
            continue
        node_id = matcher.match(record.action, graph)
        if node_id is None:
            if not getattr(matcher, "unmatched_is_neutral", False):
                off_plan.append((index, record.action))
            continue
        last_attempt[node_id] = index
        if not matcher.failed(record.observation):
            verified_at.setdefault(node_id, index)
            last_progress = index
    reached = set(last_attempt)
    stack = list(reached)
    while stack:
        for parent in predecessors[stack.pop()]:
            if parent not in reached:
                reached.add(parent)
                stack.append(parent)
    satisfied: list[PlanEdge] = []
    missing: list[tuple[str, str]] = []
    for edge in graph.edges:
        if edge.target not in last_attempt:
            continue
        if edge.source in verified_at and verified_at[edge.source] < last_attempt[edge.target]:
            satisfied.append(edge)
        else:
            missing.append((edge.source, edge.target))
    recent_off_plan = [action for index, action in off_plan if index > last_progress]
    local_nodes = [PlanNode(node_id=node.node_id, action=node.action) for node in graph.nodes if node.node_id in last_attempt]
    local_nodes += [
        PlanNode(node_id=f"off-plan-{index}", action=action[:1_000])
        for index, action in off_plan
        if index > last_progress
    ]
    return AlignmentView(
        reference_view=TaskPlan(
            nodes=[node for node in graph.nodes if node.node_id in reached],
            edges=[edge for edge in graph.edges if edge.source in reached and edge.target in reached],
        ),
        local_plan=TaskPlan(nodes=local_nodes, edges=satisfied),
        missing_prerequisites=missing,
        off_plan_actions=recent_off_plan,
        verified_nodes=frozenset(verified_at),
        step_count=len(actions),
    )
