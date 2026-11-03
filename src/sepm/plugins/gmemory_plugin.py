"""G-Memory, Generative Agents, and MetaGPT memories as host-neutral plug-ins.

The memory classes are the official ones from the G-Memory repository
(``external/GMemory/mas/memory``); this wrapper only maps the plug-in hooks to
their ``MASMemoryBase`` lifecycle (``init_task_context``, ``add_agent_node``,
``move_memory_state``, ``save_task_context``, ``retrieve_memory``,
``backward``) and formats retrieved items with G-Memory's own prompt template.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from ..llm_client import CachedEmbedder, LLMClient, UsageMeter
from ._gmemory_compat import load_gmemory

KINDS = ("empty", "generative", "metagpt", "gmemory")


class _MemoryLLM:
    """G-Memory ``LLMCallable`` charged to the ``memory`` component."""

    def __init__(self, client: LLMClient, model: str, meter: UsageMeter) -> None:
        self.client, self.model, self.meter = client, model, meter

    def __call__(self, messages: list[Any], temperature: float = 0.1, max_tokens: int = 512,
                 stop_strs: list[str] | None = None, num_comps: int = 1) -> str:
        payload = [{"role": item.role, "content": item.content} for item in messages]
        # gpt-5 family counts reasoning against the cap; same 4x headroom as execution calls.
        return self.client.chat(
            self.model, payload, meter=self.meter, component="memory_curation",
            temperature=temperature, max_tokens=max(4 * max_tokens, 8192),
        )


class GMemoryFamilyPlugin:
    def __init__(
        self,
        kind: str,
        *,
        client: LLMClient,
        meter: UsageMeter,
        model: str = "gpt-5-mini",
        gmemory_root: str | Path = "external/GMemory",
        working_dir: str | Path | None = None,
        successful_topk: int = 5,
        failed_topk: int = 1,
        insights_topk: int = 3,
        context_budget_chars: int = 12_000,
    ) -> None:
        if kind not in KINDS:
            raise ValueError(f"unknown G-Memory family kind {kind!r}")
        self.kind = kind
        self.name = {"empty": "no-memory", "generative": "generative", "metagpt": "metagpt", "gmemory": "gmemory"}[kind]
        self.classes = load_gmemory(gmemory_root)
        self.embedder = CachedEmbedder(client, meter)
        self.working_dir = str(working_dir or tempfile.mkdtemp(prefix=f"gmem-{kind}-"))
        memory_cls = self.classes[kind]
        self.memory = memory_cls(
            namespace=kind,
            global_config={"working_dir": self.working_dir, "hop": 1},
            llm_model=_MemoryLLM(client, model, meter),
            embedding_func=self.embedder,
        )
        self.successful_topk = successful_topk
        self.failed_topk = failed_topk
        self.insights_topk = insights_topk
        self.context_budget_chars = context_budget_chars
        self.counters: dict[str, float] = {}
        self._active = False
        self._retrieved: dict[str, str] = {}

    def begin_task(self, task_id: str, goal: str, description: str = "") -> None:
        self.memory.init_task_context(goal, description or goal)
        self._active = True
        self._retrieved = {}

    def context(self, agent_id: str, query: str) -> str:
        if self.kind == "empty":
            return ""
        if query in self._retrieved:
            return self._retrieved[query]
        successful, _, insights = self.memory.retrieve_memory(
            query_task=query,
            successful_topk=self.successful_topk,
            failed_topk=self.failed_topk,
            insight_topk=self.insights_topk,
            threshold=0,
        )
        self._bump("retrievals")
        self._bump("retrieved_items", len(successful) + len(insights))
        shots = [
            f"Task {index}:\n### Task description:\n{traj.task_description}\n\n"
            f"### Key steps:\n{traj.get_extra_field('key_steps')}\n\n"
            f"### Detailed trajectory:\n{traj.task_trajectory}"
            for index, traj in enumerate(successful, 1)
        ]
        rules = "\n".join(f"{index}. {rule}" for index, rule in enumerate(insights[: self.insights_topk], 1))
        if not shots and not rules:
            text = ""
        else:
            text = (
                "## Your Own Past Successes (Execution Patterns)\n" + "\n\n".join(shots)
                + "\n---\n## Key Insights from Related Tasks\n" + rules
            )[: self.context_budget_chars]
        self._retrieved[query] = text
        return text

    def record(self, agent_id: str, action: str, observation: str, **_: Any) -> list[str]:
        message = self.classes["AgentMessage"](agent_name=agent_id, message=action)
        self.memory.add_agent_node(message, upstream_agent_ids=[])
        self.memory.move_memory_state(action, observation, reward=0)
        return []

    def notices(self, agent_id: str) -> list[str]:
        return []

    def end_task(self, *, success: bool, reward: float) -> None:
        if not self._active:
            return
        self.memory.save_task_context(label=bool(success))
        self.memory.backward(bool(success))
        self._active = False

    def abandon_task(self) -> None:
        self._active = False

    def stats(self) -> dict[str, Any]:
        return dict(self.counters)

    def _bump(self, key: str, amount: float = 1.0) -> None:
        self.counters[key] = self.counters.get(key, 0.0) + amount
