"""Host-neutral memory plug-in contract.

A host (an AutoGen-style team, MARBLE's engine, ...) drives one task at a time
through five hooks and never needs to know which memory is installed:

``begin_task -> context -> record* -> notices -> end_task``

``context`` returns the text block a memory injects into an agent prompt,
``record`` writes one executed step (action and the observation it produced),
and ``notices`` returns pending corrections for the agent's next model call;
only memories with alignment (SEPM) produce notices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class MemoryPlugin(Protocol):
    name: str

    def begin_task(self, task_id: str, goal: str, description: str = "") -> None: ...

    def context(self, agent_id: str, query: str) -> str: ...

    def record(self, agent_id: str, action: str, observation: str, **kwargs: Any) -> list[str]: ...

    def notices(self, agent_id: str) -> list[str]: ...

    def end_task(self, *, success: bool, reward: float) -> None: ...

    def abandon_task(self) -> None:
        """Close a task whose outcome the host never observes (no memory update)."""

    def stats(self) -> dict[str, Any]: ...


@dataclass
class NoMemory:
    """Host without added memory; every hook is a no-op."""

    name: str = "no-memory"
    counters: dict[str, float] = field(default_factory=dict)

    def begin_task(self, task_id: str, goal: str, description: str = "") -> None:
        return None

    def context(self, agent_id: str, query: str) -> str:
        return ""

    def record(self, agent_id: str, action: str, observation: str, **_: Any) -> list[str]:
        return []

    def notices(self, agent_id: str) -> list[str]:
        return []

    def end_task(self, *, success: bool, reward: float) -> None:
        return None

    def abandon_task(self) -> None:
        return None

    def stats(self) -> dict[str, Any]:
        return dict(self.counters)
