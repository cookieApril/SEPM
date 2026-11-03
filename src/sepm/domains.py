"""Benchmark-specific rules that let the generic SEPM episode read a host's actions.

A domain supplies the verified context facts used for retrieval (Eq. 3), an
action matcher for blackboard alignment, and the world-state facts that
environment observations establish or that an agent's actions presuppose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .alignment import ActionMatcher, LexicalActionMatcher

StateReport = tuple[str, Any]


@dataclass(frozen=True)
class TaskDomain:
    """Default rules: one context fact per task family and no tracked world state."""
    task_family: str
    matcher: ActionMatcher = field(default_factory=LexicalActionMatcher)

    def context_facts(self, task_goal: str) -> set[str]:
        return {f"task_family:{self.task_family}"}

    def initial_state(self) -> list[StateReport]:
        return []

    def state_reports(self, action: str, observation: str, failed: bool) -> tuple[list[StateReport], list[StateReport]]:
        """Return (facts the environment observed, facts the agent's action presupposed)."""
        return [], []


ALFWORLD_MATCHER = LexicalActionMatcher(
    neutral_verbs=frozenset({"go", "open", "close", "look", "inventory", "examine", "think"}),
    failure_markers=("nothing happens",),
    verb_aliases={
        "pick": "take", "grab": "take", "get": "take", "place": "put", "move": "put",
        "wash": "clean", "rinse": "clean", "warm": "heat", "chill": "cool",
        "turn": "use", "switch": "use",
    },
)


def _object(text: str) -> str:
    return re.sub(r"^the\s+", "", text.strip().lower())


@dataclass(frozen=True)
class ALFWorldDomain(TaskDomain):
    """ALFWorld task types as context facts and the held object as tracked state."""
    task_family: str = "alfworld"
    matcher: ActionMatcher = ALFWORLD_MATCHER

    TASK_TYPES = (
        (re.compile(r"\btwo\b"), "pick_two_obj_and_place"),
        (re.compile(r"\bclean\b"), "pick_clean_then_place_in_recep"),
        (re.compile(r"\b(?:hot|heat)\b"), "pick_heat_then_place_in_recep"),
        (re.compile(r"\b(?:cool|cold)\b"), "pick_cool_then_place_in_recep"),
        (re.compile(r"\b(?:desklamp|examine|look at)\b"), "look_at_obj_in_light"),
    )
    PICK = re.compile(r"you pick up the (.+?) from the ")
    PUT = re.compile(r"you (?:put|move) the (.+?) (?:in|on|to|in/on) the ")
    HELD_OBJECT_ACTION = re.compile(r"^(?:put|move|clean|heat|cool|slice)\s+(.+?)\s+(?:in/on|in|on|to|with)\s+")

    def context_facts(self, task_goal: str) -> set[str]:
        lower = task_goal.lower()
        task_type = next((name for pattern, name in self.TASK_TYPES if pattern.search(lower)), "pick_and_place_simple")
        return {f"alfworld_task_type:{task_type}"}

    def initial_state(self) -> list[StateReport]:
        return [("holding", "nothing")]

    def state_reports(self, action: str, observation: str, failed: bool) -> tuple[list[StateReport], list[StateReport]]:
        lower = observation.lower()
        if picked := self.PICK.search(lower):
            held = _object(picked.group(1))
            return [("holding", held)], [("holding", held)]
        if self.PUT.search(lower):
            return [("holding", "nothing")], [("holding", "nothing")]
        presupposed = self.HELD_OBJECT_ACTION.search(action.strip().lower())
        if failed and presupposed:
            # The agent acted as if it held this object; the environment disagreed.
            return [], [("holding", _object(presupposed.group(1)))]
        return [], []


# Feedback forms used by ImplicitMemBench learning phases to signal that the
# preceding assistant turn failed (fixed from the item format, not from outcomes).
IMPLICIT_MEMORY_FAILURE_MARKERS = (
    "error", "warning", "alert", "denied", "not helpful", "unsupported", "exceeded",
    "busy", "queue full", "overloaded", "lagging", "congested", "saturated", "read-only",
    "maintenance window", "pressure high", "confus", "don't understand", "too complex",
    "failed", "fails", "cue detected", "user feedback", "impatien", "rate limit",
)


@dataclass(frozen=True)
class TextDomain(TaskDomain):
    """Free-text hosts: one context fact per task family plus caller-supplied facts."""
    extra_facts: frozenset[str] = frozenset()

    def context_facts(self, task_goal: str) -> set[str]:
        return {f"task_family:{self.task_family}", *self.extra_facts}


def text_domain(
    task_family: str,
    embedder: Any,
    *,
    failure_markers: tuple[str, ...] = (),
    extra_facts: set[str] | None = None,
    threshold: float = 0.5,
) -> TextDomain:
    from .alignment import EmbeddingActionMatcher

    return TextDomain(
        task_family=task_family,
        matcher=EmbeddingActionMatcher(embedder, failure_markers, threshold),
        extra_facts=frozenset(extra_facts or ()),
    )


def domain_for(task_family: str) -> TaskDomain:
    """Return the rules for a benchmark task family."""
    return ALFWorldDomain() if task_family == "alfworld" else TaskDomain(task_family)
