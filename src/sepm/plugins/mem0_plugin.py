"""Mem0 (automatic memory writing) as a host-neutral plug-in.

Uses the official ``mem0ai`` library: completed episodes are added as message
lists and Mem0's own LLM extracts and updates memories; ``context`` returns the
top-K memories from ``Memory.search``. Mem0's internal OpenAI clients are
wrapped so their tokens are charged to the ``memory_curation`` component.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..llm_client import LLMClient, UsageMeter


def _meter_client(openai_client: Any, meter_ref: list[UsageMeter], component: str) -> None:
    """Wrap ``chat.completions.create``/``embeddings.create`` of an OpenAI client to count usage."""
    targets = []
    if hasattr(openai_client, "chat"):
        targets.append(openai_client.chat.completions)
    if hasattr(openai_client, "embeddings"):
        targets.append(openai_client.embeddings)
    for target in targets:
        original = target.create

        def create(*args: Any, __original: Any = original, __target: Any = target, **kwargs: Any) -> Any:
            response = __original(*args, **kwargs)
            usage = getattr(response, "usage", None)
            name = "embedding" if __target.__class__.__name__.lower().startswith("embedding") else component
            meter_ref[0].add(
                name,
                int(getattr(usage, "prompt_tokens", 0) or 0),
                int(getattr(usage, "completion_tokens", 0) or 0),
            )
            return response

        target.create = create


class Mem0Plugin:
    def __init__(
        self,
        *,
        client: LLMClient,
        meter: UsageMeter,
        model: str = "gpt-5-mini",
        path: str | Path,
        user_id: str,
        top_k: int = 5,
        context_budget_chars: int = 12_000,
    ) -> None:
        os.environ.setdefault("MEM0_TELEMETRY", "False")
        from mem0 import Memory

        self.name = "mem0"
        self.user_id = user_id
        self.top_k = top_k
        self.context_budget_chars = context_budget_chars
        self.meter_ref = [meter]
        base_url = os.environ["OPENAI_BASE_URL"]
        api_key = os.environ["OPENAI_API_KEY"]
        config = {
            "llm": {"provider": "openai", "config": {
                "model": model, "api_key": api_key, "openai_base_url": base_url, "temperature": 0.0,
                "max_tokens": 16384,
            }},
            "embedder": {"provider": "openai", "config": {
                "model": "text-embedding-3-small", "api_key": api_key, "openai_base_url": base_url,
            }},
            "vector_store": {"provider": "qdrant", "config": {
                "collection_name": "mem0", "path": str(Path(path) / "qdrant"), "embedding_model_dims": 1536,
                "on_disk": False,
            }},
            "history_db_path": str(Path(path) / "history.db"),
        }
        self.memory = Memory.from_config(config)
        for holder in (getattr(self.memory, "llm", None), getattr(self.memory, "embedding_model", None)):
            inner = getattr(holder, "client", None)
            if inner is not None:
                _meter_client(inner, self.meter_ref, "memory_curation")
        self.counters: dict[str, float] = {}
        self._messages: list[dict[str, str]] = []
        self._cache: dict[str, str] = {}

    def meter_override(self, meter: UsageMeter) -> None:
        self.meter_ref[0] = meter

    def begin_task(self, task_id: str, goal: str, description: str = "") -> None:
        self._messages = [{"role": "user", "content": goal}]
        self._cache = {}

    def context(self, agent_id: str, query: str) -> str:
        if query in self._cache:
            return self._cache[query]
        found = self.memory.search(query, top_k=self.top_k, filters={"user_id": self.user_id})
        rows = found.get("results", found) if isinstance(found, dict) else found
        memories = [str(row.get("memory", "")) for row in rows if isinstance(row, dict) and row.get("memory")]
        self.counters["retrieved_items"] = self.counters.get("retrieved_items", 0.0) + len(memories)
        text = ("Relevant memories:\n" + "\n".join(f"- {item}" for item in memories))[: self.context_budget_chars] if memories else ""
        self._cache[query] = text
        return text

    def record(self, agent_id: str, action: str, observation: str, **_: Any) -> list[str]:
        self._messages.append({"role": "assistant", "content": action})
        if observation:
            self._messages.append({"role": "user", "content": observation})
        return []

    def notices(self, agent_id: str) -> list[str]:
        return []

    def end_task(self, *, success: bool, reward: float) -> None:
        if len(self._messages) > 1:
            self.memory.add(self._messages, user_id=self.user_id)
        self._messages = []

    def abandon_task(self) -> None:
        self._messages = []

    def stats(self) -> dict[str, Any]:
        return dict(self.counters)
