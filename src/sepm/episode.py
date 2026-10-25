"""One task's SEPM lifecycle inside a host loop (Algorithm 1).

``retrieve`` instantiates the top compatible SOP as the task graph, or, when no
active SOP applies, a validated candidate as a reproduction trial. Every
``record_step`` writes the action, its observation, and derived state facts to
the blackboard, then aligns the acting agent against the task graph and queues
corrections for its next model call. ``finish`` closes the episode:

* a trial is recorded against the unassisted baseline of the same task type,
  and Eq. 9 decides promotion;
* an unassisted success is distilled into a candidate whose steps and edges
  must cite verified trajectory steps (Eq. 8);
* a failure while following an active SOP attaches evidence-backed
  conditional warnings to it (Appendix B.5, B.6).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol

from .alignment import ActionRecord, derive_alignment
from .domains import TaskDomain
from .models import (
    AgentProfile,
    BlackboardEntry,
    BlackboardKind,
    CandidateStatus,
    ConditionalWarning,
    Evidence,
    EvidenceTier,
    PlanEdge,
    ProcedureGraph,
    ProcedureStep,
    ProposalOperation,
    ReproductionTrial,
    SOPCandidate,
    SOPMetadata,
    SOPVersion,
    Workspace,
    utc_now,
)
from .prompts import CANDIDATE_EXTRACTION_PROMPT, SOP_AGENT_SYSTEM_PROMPT, WARNING_EXTRACTION_PROMPT
from .service import SEPMService
from .storage import ConcurrencyError

ENVIRONMENT_AGENT = "environment"


class SOPModel(Protocol):
    """LLM used for procedure extraction and warnings; returns one parsed JSON object."""
    def complete_json(self, system: str, user: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class EpisodeStep:
    agent_id: str
    action: str
    observation: str
    failed: bool


@dataclass(frozen=True)
class EpisodeContext:
    """Procedures injected into this episode; ``reference`` is the instantiated task graph."""
    active: list[SOPVersion] = field(default_factory=list)
    trial: SOPVersion | None = None

    @property
    def reference(self) -> SOPVersion | None:
        return self.active[0] if self.active else self.trial


def string_list(value: Any, field_name: str) -> list[str]:
    """Accept a string or a list from model JSON and return a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a string or list")
    return [str(item) for item in value]


def format_procedure(sop: SOPVersion) -> str:
    """Render steps, prerequisite edges, and conditional warnings for an agent prompt."""
    labels = {step.step_id: step.instruction for step in sop.procedure.steps}
    lines = [f"{index}. {step.instruction}" for index, step in enumerate(sop.procedure.steps, 1)]
    lines += [f"Prerequisite: {labels[edge.source]!r} before {labels[edge.target]!r}" for edge in sop.procedure.edges]
    lines += [
        f"Warning: {warning.text}" + (f" (when: {warning.trigger_condition})" if warning.trigger_condition else "")
        for warning in sop.metadata.warnings
    ]
    return "\n".join(lines)


def _cited_steps(value: Any, step_count: int) -> list[int]:
    """Return valid 1-based step citations, or an empty list if any citation is invalid."""
    if not isinstance(value, list) or not value:
        return []
    if not all(isinstance(item, int) and not isinstance(item, bool) and 1 <= item <= step_count for item in value):
        return []
    return list(value)


class ProceduralEpisode:
    """SEPM state for one task; durable state lives in the service's SQLite store."""

    def __init__(
        self,
        service: SEPMService,
        domain: TaskDomain,
        *,
        workspace_id: str,
        task_goal: str,
        sop_model: SOPModel | None = None,
        stats: dict[str, float] | None = None,
        max_steps: int = 30,
        max_candidate_trials: int = 3,
        min_trial_adoption: float = 0.5,
        trial_step_cost: float = 0.0,
        max_warnings: int = 3,
    ) -> None:
        self.service = service
        self.domain = domain
        self.task_family = domain.task_family
        self.workspace_id = workspace_id
        self.task_goal = task_goal
        self.sop_model = sop_model
        self.stats = stats if stats is not None else {}
        self.max_steps = max_steps
        self.max_candidate_trials = max_candidate_trials
        self.min_trial_adoption = min_trial_adoption
        self.trial_step_cost = trial_step_cost
        self.max_warnings = max_warnings
        self.context_facts = domain.context_facts(task_goal)
        self.context_key = "|".join(sorted(self.context_facts)) or self.task_family
        self.steps: list[EpisodeStep] = []
        self.context: EpisodeContext | None = None
        self.finished = False
        self._agents: set[str] = set()
        self._notices: list[str] = []
        self._last_signature: dict[str, tuple[str, ...]] = {}
        self._last_observed: datetime | None = None
        service.create_workspace(Workspace(workspace_id=workspace_id, main_goal=task_goal[:10_000]))
        self.register_agent(ENVIRONMENT_AGENT, "official environment observations")
        for key, value in domain.initial_state():
            self._state(ENVIRONMENT_AGENT, key, value, observed=True)

    # ---------- Retrieval and instantiation ----------
    def retrieve(self, query: str, limit: int = 5) -> EpisodeContext:
        """Retrieve once per episode and instantiate the reference procedure as the task graph."""
        if self.context is not None:
            return self.context
        active = [
            result.sop
            for result in self.service.retrieve_sops(query, limit, context_facts=self.context_facts)
        ]
        trial = None
        if not active:
            ranked = self.service.retrieve_trial_candidates(
                query, 1, context_facts=self.context_facts, task_family=self.task_family
            )
            trial = ranked[0].sop if ranked else None
        self.context = EpisodeContext(active=active, trial=trial)
        if self.context.reference is not None:
            self.service.instantiate_plan(self.workspace_id, self.context.reference.procedure)
        self._bump("sop_retrieved_episodes", bool(active))
        self._bump("sop_trial_episodes", trial is not None)
        return self.context

    # ---------- Blackboard recording and alignment ----------
    def register_agent(self, agent_id: str, role: str | None = None) -> None:
        if agent_id not in self._agents:
            self.service.register_agent(AgentProfile(agent_id=agent_id, role=(role or agent_id)[:512]))
            self._agents.add(agent_id)

    def record_step(self, agent_id: str, action: str, observation: str) -> list[str]:
        """Write one step to the blackboard, align the agent, and return new corrections."""
        self.register_agent(agent_id)
        failed = self.domain.matcher.failed(observation)
        self.steps.append(EpisodeStep(agent_id, action, observation, failed))
        self.service.append_blackboard(
            BlackboardEntry(
                workspace_id=self.workspace_id,
                agent_id=agent_id,
                kind=BlackboardKind.ACTION,
                action=action[:4_000],
                observation=observation[:8_000],
                evidence=[
                    Evidence(
                        tier=EvidenceTier.TOOL_OBSERVATION,
                        source=f"{self.task_family}:environment",
                        value=observation[:2_000],
                        observed_at=self._observed_at(),
                    )
                ],
            )
        )
        observed, presupposed = self.domain.state_reports(action, observation, failed)
        for key, value in observed:
            self._state(ENVIRONMENT_AGENT, key, value, observed=True)
        for key, value in presupposed:
            self._state(agent_id, key, value, observed=False)
        result = self.service.align_agent(self.workspace_id, agent_id, self.domain.matcher)
        self._bump("alignment_checks")
        signature = tuple(result.notices)
        if signature == self._last_signature.get(agent_id, ()):
            return []
        self._last_signature[agent_id] = signature
        if result.report.requires_alignment:
            self._bump("alignment_events")
            for reason in result.report.reasons:
                self._bump(f"alignment_{reason}")
        self._notices.extend(result.notices)
        return list(result.notices)

    def take_notices(self) -> list[str]:
        """Hand queued corrections to the next model call exactly once."""
        notices, self._notices = self._notices, []
        self._bump("alignment_notices_delivered", amount=len(notices))
        return notices

    def _observed_at(self) -> datetime:
        """Strictly increasing observation times, so the latest same-source report wins (Appendix B.3)."""
        now = utc_now()
        if self._last_observed is not None and now <= self._last_observed:
            now = self._last_observed + timedelta(microseconds=1)
        self._last_observed = now
        return now

    def _state(self, agent_id: str, key: str, value: Any, *, observed: bool) -> None:
        # Environment facts carry tool evidence; presupposed facts are the agent's inference.
        self.service.append_blackboard(
            BlackboardEntry(
                workspace_id=self.workspace_id,
                agent_id=agent_id,
                kind=BlackboardKind.OBSERVATION if observed else BlackboardKind.STATE_CLAIM,
                state_key=key,
                state_value=value,
                evidence=(
                    [Evidence(
                        tier=EvidenceTier.TOOL_OBSERVATION,
                        source=f"{self.task_family}:environment",
                        value=value,
                        observed_at=self._observed_at(),
                    )]
                    if observed
                    else []
                ),
            )
        )

    # ---------- Episode outcome and memory evolution ----------
    def finish(self, *, success: bool, reward: float) -> None:
        """Record outcomes, trials, candidates, and warnings once per episode."""
        if self.finished:
            return
        self.finished = True
        context = self.context or EpisodeContext()
        for sop in context.active:
            self.service.record_sop_outcome(sop.sop_id, success)
        if context.trial is not None:
            self._record_trial(context.trial, success=success, reward=reward)
        elif not context.active:
            self._add_baseline(reward)
        if self.sop_model is None or not self.service.config.enable_procedural_memory:
            return
        if success and not context.active and context.trial is None:
            self._propose_candidate(reward)
        elif not success and context.active:
            self._attach_failure_warnings(context.active[0])

    def baseline_reward(self) -> float | None:
        """Mean reward of unassisted episodes of this task type, else of the whole run."""
        for scope in (self.context_key, "all"):
            count = self.stats.get(f"baseline_episodes::{scope}", 0.0)
            if count:
                return self.stats.get(f"baseline_reward_sum::{scope}", 0.0) / count
        return None

    def _add_baseline(self, reward: float) -> None:
        for scope in (self.context_key, "all"):
            self._bump(f"baseline_episodes::{scope}")
            self._bump(f"baseline_reward_sum::{scope}", amount=reward)

    def _record_trial(self, candidate: SOPVersion, *, success: bool, reward: float) -> None:
        baseline = self.baseline_reward()
        if baseline is None:
            # Without an unassisted baseline the trial cannot measure net benefit (Eq. 13).
            self._bump("sop_trials_deferred")
            return
        view = derive_alignment(
            candidate.procedure.as_plan(),
            [ActionRecord(step.action, step.observation) for step in self.steps],
            self.domain.matcher,
        )
        # Attribution: success counts for the candidate only if its steps were actually executed.
        adoption = len(view.verified_nodes) / view.step_count if view.step_count else 0.0
        self.service.record_reproduction(
            ReproductionTrial(
                candidate_id=candidate.sop_id,
                workspace_id=self.workspace_id,
                task_family=self.task_family,
                environment_fingerprint=f"{self.context_key}#{self.workspace_id}",
                success=success,
                baseline_reward=baseline,
                candidate_reward=reward,
                cost=self.trial_step_cost * len(self.steps) / max(1, self.max_steps),
                state_verified=True,
                causal_supported=success and adoption >= self.min_trial_adoption,
            )
        )
        self._bump("sop_trials_recorded")
        status = self.service.store.get_candidate(candidate.sop_id).status
        if status == CandidateStatus.PROMOTED:
            self._bump("sop_promotions")
        elif len(self.service.store.list_trials(candidate.sop_id)) >= self.max_candidate_trials:
            self.service.store.update_candidate_status(candidate.sop_id, CandidateStatus.REJECTED)
            self._bump("sop_candidates_retired")

    def numbered_trajectory(self, limit: int = 80) -> str:
        return "\n".join(
            f"{index}. [{step.agent_id}] action: {step.action}\n   observation: {step.observation[:300]}"
            for index, step in enumerate(self.steps[:limit], 1)
        )

    def _propose_candidate(self, reward: float) -> None:
        prompt = CANDIDATE_EXTRACTION_PROMPT.format(task=self.task_goal, trajectory=self.numbered_trajectory())
        try:
            raw = self.sop_model.complete_json(SOP_AGENT_SYSTEM_PROMPT, prompt)
            candidate = self.candidate_from_json(raw, reward)
        except Exception as error:  # noqa: BLE001 - Memory curation must never fail the episode.
            self._bump("sop_candidate_rejections")
            print(f"SOP candidate rejected: {error}", file=sys.stderr)
            return
        stored, safety = self.service.propose_sop(candidate)
        self._bump("sop_candidates")
        if not safety.allowed:
            self._bump("sop_safety_rejections")
            return
        validated, _ = self.service.validate_candidate(stored.candidate_id)
        if validated.status == CandidateStatus.VALIDATED:
            self._bump("sop_candidates_passed")

    def candidate_from_json(self, raw: dict[str, Any], reward: float) -> SOPCandidate:
        """Keep only steps and edges the trajectory verifies (Eq. 8)."""
        raw_steps = raw["atomic_steps"]
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError("atomic_steps must be a non-empty list")
        evidence: dict[int, list[int]] = {}
        steps: list[ProcedureStep] = []
        for index, item in enumerate(raw_steps, 1):
            if not isinstance(item, dict):
                continue
            cited = _cited_steps(item.get("evidence_steps"), len(self.steps))
            # A step is verified when a cited action succeeded in the environment.
            if cited and any(not self.steps[k - 1].failed for k in cited):
                evidence[index] = cited
                steps.append(ProcedureStep(step_id=f"step-{len(steps) + 1}", instruction=str(item["instruction"])[:4_000]))
        if not steps:
            raise ValueError("no extracted step is supported by the trajectory")
        new_id = {old: f"step-{position}" for position, old in enumerate(evidence, 1)}
        edges: list[PlanEdge] = []
        for pair in raw.get("dependencies") or []:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(f"invalid dependency pair: {pair!r}")
            source, target = pair
            if not all(isinstance(end, int) and not isinstance(end, bool) for end in pair):
                raise ValueError(f"invalid dependency pair: {pair!r}")
            # Both ends must be verified, and the prerequisite must be observed before the dependent.
            if source in evidence and target in evidence and min(evidence[source]) < max(evidence[target]):
                edge = PlanEdge(source=new_id[source], target=new_id[target])
                if edge not in edges:
                    edges.append(edge)
        return SOPCandidate(
            operation=ProposalOperation.CREATE,
            procedure=ProcedureGraph(steps=steps, edges=edges),
            metadata=SOPMetadata(
                title=str(raw["title"])[:512],
                description=str(raw.get("description") or raw.get("rationale") or "")[:8_000],
                task_family=self.task_family,
                applicability=string_list(raw.get("applicability"), "applicability"),
                exclusions=string_list(raw.get("exclusions"), "exclusions"),
                tags=["sepm-extracted", self.task_family],
                context_conditions=sorted(self.context_facts),
            ),
            source_workspace_ids=[self.workspace_id],
            # The outcome comes from the official environment evaluator.
            state_verified=True,
            causal_confidence=len(steps) / len(raw_steps),
            claimed_benefit=reward,
        )

    def _attach_failure_warnings(self, sop: SOPVersion) -> None:
        prompt = WARNING_EXTRACTION_PROMPT.format(
            max_warnings=self.max_warnings,
            task=self.task_goal,
            procedure=format_procedure(sop),
            trajectory=self.numbered_trajectory(),
        )
        try:
            raw = self.sop_model.complete_json(SOP_AGENT_SYSTEM_PROMPT, prompt)
            items = raw["warnings"]
            if not isinstance(items, list):
                raise TypeError("warnings must be a list")
        except Exception as error:  # noqa: BLE001 - Memory curation must never fail the episode.
            self._bump("sop_warning_rejections")
            print(f"SOP warnings rejected: {error}", file=sys.stderr)
            return
        warnings: list[ConditionalWarning] = []
        for item in items[: self.max_warnings]:
            if not isinstance(item, dict) or not str(item.get("text", "")).strip():
                continue
            cited = _cited_steps(item.get("evidence_steps"), len(self.steps))
            if not cited:
                # A lesson without trajectory evidence is not supported.
                continue
            warnings.append(
                ConditionalWarning(
                    text=str(item["text"]).strip()[:2_000],
                    trigger_condition=str(item.get("trigger_condition") or "").strip()[:2_000],
                    evidence=[f"{self.workspace_id}#step-{k}" for k in cited],
                )
            )
        if not warnings:
            return
        try:
            head = self.service.attach_warnings(sop, warnings)
        except (ConcurrencyError, ValueError):
            # The SOP's structure changed or it retired while this episode ran; its head stays unchanged.
            self._bump("sop_warning_conflicts")
            return
        change = head.change
        if change is not None and change.kind == "warnings" and change.base_version == sop.version:
            self._bump("sop_warning_revisions")
            self._bump("sop_warning_rebases", change.rebased_onto is not None)

    def _bump(self, key: str, condition: bool = True, *, amount: float = 1.0) -> None:
        if condition:
            self.stats[key] = self.stats.get(key, 0.0) + amount
