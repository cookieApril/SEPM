"""SEPM as a host-neutral memory plug-in.

Wraps :class:`~sepm.episode.ProceduralEpisode` (Algorithm 1) behind the
:mod:`sepm.plugins.base` hooks. The SOP model curates procedures, an LLM judge
scores goal consistency (J_goal), and API embeddings back retrieval and the
free-text action matcher.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..ablation.runtime import apply_ablation
from ..config import MemoryConfig
from ..domains import TaskDomain, text_domain
from ..episode import ProceduralEpisode, render_procedure
from ..llm_client import CachedEmbedder, LLMClient, UsageMeter
from ..service import SEPMService

GOAL_JUDGE_PROMPT = """Rate how consistent an agent's stated goal is with the goal it was assigned
and with the team's overall task objective. Return exactly one JSON object
{{"score": <number between 0 and 1>}} where 1 means fully consistent and 0 means contradictory
or unrelated.

Assigned goal: {assigned}
Task objective: {objective}
Agent's stated goal: {agent}
"""


def _parse_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model response contains no JSON object")
    value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise TypeError("model response must be a JSON object")
    return value


class LLMSOPModel:
    """SOP curation model returning one parsed JSON object (``memory`` component)."""

    def __init__(self, client: LLMClient, model: str, meter: UsageMeter) -> None:
        self.client, self.model, self.meter = client, model, meter

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        text = self.client.chat(
            self.model,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            meter=self.meter,
            component="memory_curation",
            temperature=0.0,
            max_tokens=16_384,
            json_mode=True,
        )
        return _parse_json(text)


class LLMGoalJudge:
    """Frozen LLM goal-consistency judge J_goal (``divergence`` component)."""

    def __init__(self, client: LLMClient, model: str, meter: UsageMeter) -> None:
        self.client, self.model, self.meter = client, model, meter

    def consistency_score(self, agent_goal: str, assigned_goal: str, task_objective: str) -> float:
        text = self.client.chat(
            self.model,
            [{"role": "user", "content": GOAL_JUDGE_PROMPT.format(
                assigned=assigned_goal[:2_000], objective=task_objective[:2_000], agent=agent_goal[:2_000]
            )}],
            meter=self.meter,
            component="divergence_judge",
            temperature=0.0,
            max_tokens=4096,
            json_mode=True,
        )
        try:
            return max(0.0, min(1.0, float(_parse_json(text)["score"])))
        except (KeyError, TypeError, ValueError):
            return 1.0  # An unreadable verdict must not trigger realignment.


class SEPMPlugin:
    """One SEPM procedure pool; tasks run one at a time against it."""

    def __init__(
        self,
        *,
        database_path: str | Path,
        task_family: str,
        client: LLMClient,
        meter: UsageMeter,
        model: str = "gpt-5-mini",
        judge_model: str = "gpt-5-mini",
        ablation: str = "full",
        config_overrides: dict[str, Any] | None = None,
        failure_markers: tuple[str, ...] = (),
        extra_facts: set[str] | None = None,
        match_threshold: float = 0.5,
        top_k: int = 5,
        max_steps: int = 30,
        domain: TaskDomain | None = None,
        textual_procedures: bool = False,
    ) -> None:
        self.name = "sepm"
        config = apply_ablation(MemoryConfig(database_path=Path(database_path)), ablation)
        if config_overrides:
            config = replace(config, **config_overrides)
        self.config = config
        self.meter = meter
        self.embedder = CachedEmbedder(client, meter)
        self.service = SEPMService(
            config=config,
            embedder=self.embedder,
            goal_judge=LLMGoalJudge(client, judge_model, meter),
        )
        self.sop_model = LLMSOPModel(client, model, meter)
        self.domain = domain or text_domain(
            task_family, self.embedder, failure_markers=failure_markers,
            extra_facts=extra_facts, threshold=match_threshold,
        )
        self.top_k = top_k
        self.max_steps = max_steps
        self.textual_procedures = textual_procedures
        self.counters: dict[str, float] = {}
        self.last_context_ids: list[str] = []
        self.episode: ProceduralEpisode | None = None
        self.task_counters: list[dict[str, float]] = []

    # ---------- hooks ----------
    def begin_task(self, task_id: str, goal: str, description: str = "") -> None:
        if self.episode is not None and not self.episode.finished:
            raise RuntimeError("previous task was not ended")
        self.episode = ProceduralEpisode(
            self.service,
            self.domain,
            workspace_id=f"{task_id}-{uuid.uuid4().hex[:8]}",
            task_goal=goal[:10_000],
            sop_model=self.sop_model,
            stats=self.counters,
            max_steps=self.max_steps,
        )
        self._task_start = dict(self.counters)

    def context(self, agent_id: str, query: str) -> str:
        episode = self._require()
        self.episode.register_agent(agent_id)
        retrieved = episode.retrieve(query, self.top_k)
        self.last_context_ids = [sop.sop_id for sop in retrieved.active] + (
            [retrieved.trial.sop_id] if retrieved.trial is not None else []
        )
        blocks: list[str] = []
        for sop in retrieved.active:
            blocks.append(f"Procedure: {sop.metadata.title}\n{self._render(sop)}")
        if retrieved.trial is not None:
            blocks.append(
                "Candidate procedure under validation; verify each step.\n"
                f"Procedure: {retrieved.trial.metadata.title}\n{self._render(retrieved.trial)}"
            )
        if not blocks:
            return ""
        return "Shared team procedures (task graph with prerequisites and conditional warnings):\n\n" + "\n\n".join(blocks)

    def assign(self, agent_id: str, goal: str) -> None:
        """Manager assignment: binds an agent's reference goal on the shared task state."""
        episode = self._require()
        episode.register_agent(agent_id)
        self.service.assign_agent(episode.workspace_id, agent_id, goal=goal[:10_000])

    def record(self, agent_id: str, action: str, observation: str, *, goal: str | None = None) -> list[str]:
        if not (self.config.share_blackboard_across_agents or self.config.enable_procedural_memory
                or self.config.enable_divergence_detection):
            return []
        return self._require().record_step(agent_id, action, observation, goal=goal)

    def notices(self, agent_id: str) -> list[str]:
        return self.episode.take_notices() if self.episode is not None else []

    def end_task(self, *, success: bool, reward: float) -> None:
        episode = self._require()
        episode.finish(success=success, reward=reward)
        self.task_counters.append(
            {key: value - self._task_start.get(key, 0.0) for key, value in self.counters.items()}
        )

    def abandon_task(self) -> None:
        """Close a task without an observed outcome: no outcome, trial, or extraction."""
        if self.episode is not None:
            self.episode.finished = True

    def stats(self) -> dict[str, Any]:
        return dict(self.counters)

    def pool_statistics(self) -> dict[str, Any]:
        """Procedure-pool statistics of Supp. Table S13a read from the store."""
        from ..models import CandidateStatus

        store = self.service.store
        counts = {}
        for status in CandidateStatus:
            _, total = store.list_candidates(status, limit=1, offset=0)
            counts[status.value] = total
        active, active_total = store.list_active_sops(limit=1_000, offset=0)
        return {
            "candidates_by_status": counts,
            "active_procedures": active_total,
            "avg_versions_per_active_procedure": (
                sum(sop.version for sop in active) / len(active) if active else 0.0
            ),
            "warnings_on_active": sum(len(sop.metadata.warnings) for sop in active),
            "specialized_variants": sum(1 for sop in active if sop.metadata.variant_of),
        }

    # ---------- helpers ----------
    def _render(self, sop: Any) -> str:
        return render_procedure(sop, self.config.explicit_task_graph and not self.textual_procedures)

    def _require(self) -> ProceduralEpisode:
        if self.episode is None:
            raise RuntimeError("begin_task must be called first")
        return self.episode
