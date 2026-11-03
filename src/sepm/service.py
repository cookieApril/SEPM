"""Business orchestration layer and recommended facade for SEPM.

The service connects storage, divergence detection, evidence resolution, safety,
SOP promotion, and retrieval into one workflow. Framework adapters, the CLI,
benchmarks, tests, and examples reuse it instead of duplicating rules.

The publication path is propose_sop -> validate_candidate ->
record_reproduction. The first qualified successful trial immediately calls
promote_candidate, which rechecks every gate of Eq. 9 and commits through
storage CAS. Conditional warnings are attached separately by attach_warnings
and do not require a structural revision.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from statistics import fmean
from typing import Any

from .alignment import ActionMatcher, ActionRecord, derive_alignment
from .config import MemoryConfig
from .divergence import DivergenceDetector, GoalJudge
from .embedding import Embedder, HashingEmbedder
from .evidence import EvidenceResolver
from .models import (
    AgentAssignment,
    AgentProfile,
    AlignmentResult,
    BlackboardEntry,
    BlackboardKind,
    CandidateStatus,
    ConditionalWarning,
    DivergenceReport,
    Evidence,
    EvidenceTier,
    ProcedureGraph,
    PromotionDecision,
    ProposalOperation,
    ReproductionTrial,
    Resolution,
    RetrievalResult,
    SafetyDecision,
    SOPCandidate,
    SOPChange,
    SOPVersion,
    Workspace,
    procedure_fingerprint,
    utc_now,
)
from .retrieval import AdaptiveWeightRouter, SOPRetriever
from .safety import SafetyValidator
from .storage import ConcurrencyError, NotFoundError, SQLiteStore


def _canonical(value: Any) -> str:
    """Serialize state values stably so key order does not affect conflict detection."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _merge_warnings(
    existing: list[ConditionalWarning], incoming: list[ConditionalWarning]
) -> tuple[list[ConditionalWarning], list[ConditionalWarning]]:
    """Append incoming lessons not already present; returns (merged, added)."""
    known = {(item.text.casefold(), item.trigger_condition.casefold()) for item in existing}
    added: list[ConditionalWarning] = []
    for item in incoming:
        key = (item.text.casefold(), item.trigger_condition.casefold())
        if key not in known:
            known.add(key)
            added.append(item)
    return [*existing, *added], added


class SEPMService:
    """Shared facade with injectable dependencies for tests and experiments."""

    # The five promotion gates of Eq. 9.
    PROMOTION_GATES = ("safety", "state_verified", "causal_attribution", "reproduction", "net_benefit")
    # Rebase-and-retry rounds before a concurrent writer is treated as a conflict.
    MAX_REBASE_ATTEMPTS = 3

    def __init__(
        self,
        database_path: str | Path | None = None,
        *,
        config: MemoryConfig | None = None,
        embedder: Embedder | None = None,
        goal_judge: GoalJudge | None = None,
    ) -> None:
        """Initialize storage, detection, resolution, safety, and retrieval components."""
        self.config = config or MemoryConfig(database_path=Path(database_path or "sepm.db"))
        if database_path is not None and Path(database_path) != self.config.database_path:
            self.config = MemoryConfig(**{**self.config.__dict__, "database_path": Path(database_path)})
        self.store = SQLiteStore(
            self.config.database_path, retained_previous_versions=self.config.retained_previous_versions
        )
        self.embedder = embedder or HashingEmbedder()
        self.detector = DivergenceDetector(self.embedder, goal_judge, self.config)
        self.resolver = EvidenceResolver(self.config)
        self.safety = SafetyValidator(self.config)
        self.router = AdaptiveWeightRouter(self.config)
        self.retriever = SOPRetriever(self.embedder, self.router, self.config)

    # ---------- Private memory and shared task state ----------
    def register_agent(self, profile: AgentProfile) -> AgentProfile:
        self.store.upsert_agent(profile)
        return profile

    def create_workspace(self, workspace: Workspace) -> Workspace:
        self.store.put_workspace(workspace, overwrite=False)
        return self.store.get_workspace(workspace.workspace_id)

    def assign_agent(
        self,
        workspace_id: str,
        agent_id: str,
        *,
        goal: str | None = None,
        node_ids: list[str] | None = None,
    ) -> Workspace:
        """Bind an agent to its reference goal and task-graph nodes (manager role)."""
        self.store.get_agent(agent_id)
        workspace = self.store.get_workspace(workspace_id)
        assignments = {
            **workspace.assignments,
            agent_id: AgentAssignment(goal=goal, node_ids=list(node_ids or [])),
        }
        # Re-validate through the model so assignments must reference plan nodes.
        updated = Workspace.model_validate({**workspace.model_dump(), "assignments": assignments})
        self.store.put_workspace(updated)
        return updated

    def instantiate_plan(self, workspace_id: str, procedure: ProcedureGraph) -> Workspace:
        """Bind a retrieved procedure's steps and prerequisites as the workspace task graph."""
        workspace = self.store.get_workspace(workspace_id)
        plan = procedure.as_plan()
        known = {node.node_id for node in plan.nodes}
        updated = Workspace.model_validate({
            **workspace.model_dump(),
            "plan": plan.model_dump(),
            "plan_version": workspace.plan_version + 1,
            # Keep assigned goals; node assignments survive only where the node still exists.
            "assignments": {
                agent_id: {"goal": item.goal, "node_ids": [node for node in item.node_ids if node in known]}
                for agent_id, item in workspace.assignments.items()
            },
        })
        self.store.put_workspace(updated)
        return updated

    def append_blackboard(self, entry: BlackboardEntry) -> BlackboardEntry:
        """Append an event after validating agent/workspace; appending never mutates the goal."""
        self.store.get_agent(entry.agent_id)
        self.store.get_workspace(entry.workspace_id)
        self.store.append_entry(entry, self.config.max_blackboard_entries)
        return entry

    def list_blackboard(
        self,
        workspace_id: str,
        limit: int = 50,
        offset: int = 0,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """Read the blackboard; nonshared ablations require an agent-local view."""
        if not self.config.share_blackboard_across_agents and agent_id is None:
            raise ValueError("agent_id is required when shared blackboard is disabled")
        entries, total = self.store.list_entries(workspace_id, limit, offset, agent_id)
        return {
            "total": total,
            "count": len(entries),
            "offset": offset,
            "has_more": offset + len(entries) < total,
            "next_offset": offset + len(entries) if offset + len(entries) < total else None,
            "items": entries,
        }

    # ---------- Structured divergence and evidence-first resolution ----------
    def detect_divergence(self, workspace_id: str, agent_id: str) -> DivergenceReport:
        """Compare an agent's latest blackboard entry with canonical workspace state."""
        workspace = self.store.get_workspace(workspace_id)
        agent_entries, _ = self.store.list_entries(
            workspace_id, limit=self.config.max_blackboard_entries, offset=0, agent_id=agent_id
        )
        if not agent_entries:
            raise NotFoundError(f"no blackboard state exists for agent {agent_id!r}")
        # Events are sparse: an observation must not erase a previously reported goal/plan.
        latest = agent_entries[0].model_copy(update={
            "goal": next((entry.goal for entry in agent_entries if entry.goal is not None), None),
            "plan": next((entry.plan for entry in agent_entries if entry.plan is not None), None),
        })
        if not self.config.enable_divergence_detection:
            return self._disabled_report(agent_id)
        conflicts = self._state_conflicts(workspace_id, focus_agent=agent_id)
        return self.detector.inspect(workspace, latest, conflicts)

    def align_agent(self, workspace_id: str, agent_id: str, matcher: ActionMatcher) -> AlignmentResult:
        """Read one agent's actions from the blackboard, detect divergence, and build corrections.

        The agent's local plan is derived from its executed actions rather than
        reported by the agent (see ``alignment.derive_alignment``). Corrections name
        unverified prerequisites, off-plan actions, and verified values of disputed state.
        """
        if not self.config.enable_divergence_detection:
            return AlignmentResult(report=self._disabled_report(agent_id))
        workspace = self.store.get_workspace(workspace_id)
        entries, _ = self.store.list_entries(
            workspace_id, limit=self.config.max_blackboard_entries, offset=0, agent_id=agent_id
        )
        entries.reverse()
        view = None
        reference = workspace.reference_plan(agent_id)
        if reference.nodes:
            view = derive_alignment(reference, self._action_records(entries), matcher)
        conflicts = self._state_conflicts(workspace_id, focus_agent=agent_id)
        probe = BlackboardEntry(
            workspace_id=workspace_id,
            agent_id=agent_id,
            kind=BlackboardKind.PLAN,
            goal=next((entry.goal for entry in reversed(entries) if entry.goal is not None), None),
            plan=view.local_plan if view else None,
        )
        report = self.detector.inspect(
            workspace, probe, conflicts, reference_plan=view.reference_view if view else None
        )
        notices: list[str] = []
        if "goal_divergence" in report.reasons:
            notices.append(f"Your stated goal diverges from your assignment: {workspace.reference_goal(agent_id)!r}.")
        if "plan_divergence" in report.reasons and view is not None:
            actions = {node.node_id: node.action for node in reference.nodes}
            notices.extend(
                f"Before {actions[target][:160]!r}, complete and verify the prerequisite {actions[source][:160]!r}."
                for source, target in view.missing_prerequisites
            )
            notices.extend(
                f"{action[:160]!r} is outside the current procedure; return to its next unverified step."
                for action in view.off_plan_actions
            )
        resolved: dict[str, Any] = {}
        for key in report.conflicting_state_keys:
            resolution = self.resolve_state(
                workspace_id, key, None if self.config.share_blackboard_across_agents else agent_id
            )
            if resolution.resolved:
                resolved[key] = resolution.value
                notices.append(
                    f"Verified state {key!r} is {resolution.value!r}; act on it instead of your assumption."
                )
            else:
                notices.append(f"State {key!r} is disputed: {resolution.next_action}.")
        return AlignmentResult(
            report=report,
            missing_prerequisites=[f"{source}->{target}" for source, target in (view.missing_prerequisites if view else [])],
            off_plan_actions=list(view.off_plan_actions) if view else [],
            resolved_state=resolved,
            notices=notices,
        )

    @staticmethod
    def _action_records(entries: list[BlackboardEntry]) -> list[ActionRecord]:
        """Pair each action, oldest first, with its observation or the next result entry."""
        records: list[ActionRecord] = []
        pending: str | None = None
        for entry in entries:
            if entry.action is not None:
                if pending is not None:
                    records.append(ActionRecord(pending))
                if entry.observation is not None:
                    records.append(ActionRecord(entry.action, entry.observation))
                    pending = None
                else:
                    pending = entry.action
            elif pending is not None and (entry.result is not None or entry.observation is not None):
                records.append(ActionRecord(pending, entry.result or entry.observation or ""))
                pending = None
        if pending is not None:
            records.append(ActionRecord(pending))
        return records

    @staticmethod
    def _disabled_report(agent_id: str) -> DivergenceReport:
        return DivergenceReport(
            agent_id=agent_id,
            goal_divergence=0.0,
            goal_embedding_component=0.0,
            goal_judge_component=0.0,
            plan_divergence=0.0,
            conflicting_state_keys=[],
            requires_alignment=False,
            reasons=[],
            alignment_stage="none",
            recommended_action="divergence detection disabled",
        )

    def resolve_state(
        self,
        workspace_id: str,
        state_key: str,
        agent_id: str | None = None,
    ) -> Resolution:
        """Collect evidence for one key and resolve it by the evidence hierarchy."""
        if not self.config.share_blackboard_across_agents and agent_id is None:
            raise ValueError("agent_id is required when shared blackboard is disabled")
        entries = self.store.list_state_entries(workspace_id, state_key, agent_id=agent_id)
        evidence = [item for entry in entries for item in self._entry_evidence(entry)]
        return self.resolver.resolve_world_state(state_key, evidence)

    @staticmethod
    def _entry_evidence(entry: BlackboardEntry) -> list[Evidence]:
        """Return an entry's evidence; a bare state claim is the agent's own inference."""
        if entry.evidence:
            return list(entry.evidence)
        if entry.state_value is None:
            return []
        # Agent inference never establishes external state: the resolver refuses to
        # resolve a key whose highest qualified tier is AGENT_INFERENCE.
        return [
            Evidence(
                tier=EvidenceTier.AGENT_INFERENCE,
                source=f"agent:{entry.agent_id}",
                value=entry.state_value,
                observed_at=entry.created_at,
            )
        ]

    def _state_conflicts(self, workspace_id: str, focus_agent: str | None = None) -> list[str]:
        """Return C_s (Eq. 7): keys whose qualified evidence reports conflicting values.

        Each agent contributes the qualified evidence from its latest report on a key,
        so an agent revising its own observation is not a permanent conflict.
        """
        local_agent = focus_agent if not self.config.share_blackboard_across_agents else None
        entries = self.store.list_state_entries(workspace_id, agent_id=local_agent)
        values: dict[str, dict[str, set[str]]] = defaultdict(dict)
        for entry in entries:
            # Storage returns newest first, so the first entry seen is the agent's latest report.
            if entry.state_key is None or entry.agent_id in values[entry.state_key]:
                continue
            values[entry.state_key][entry.agent_id] = {
                _canonical(item.value)
                for item in self._entry_evidence(entry)
                if item.confidence >= self.config.state_confidence_threshold
            }
        conflicts: list[str] = []
        for key, per_agent in values.items():
            if focus_agent and focus_agent not in per_agent:
                continue
            if len(set().union(*per_agent.values())) > 1:
                conflicts.append(key)
        return sorted(conflicts)

    # ---------- Candidate queue, safety gate, and atomic promotion ----------
    def propose_sop(self, candidate: SOPCandidate) -> tuple[SOPCandidate, SafetyDecision]:
        """Precheck and persist a candidate; update/delete must identify a target version."""
        if not self.config.enable_procedural_memory:
            decision = SafetyDecision(
                allowed=False,
                risk_score=0.0,
                risk_threshold=self.config.safety_risk_threshold,
                violations=["procedural memory disabled for this experiment condition"],
            )
            candidate = candidate.model_copy(update={"status": CandidateStatus.REJECTED})
            self.store.put_candidate(candidate)
            return candidate, decision
        current = None
        if candidate.metadata and candidate.metadata.variant_of:
            parent = self.store.get_sop(candidate.metadata.variant_of)
            if not parent.active:
                raise ValueError("variant parent must be active")
        if candidate.target_sop_id:
            current = self.store.get_sop(candidate.target_sop_id)
            if candidate.base_version is None:
                raise ValueError("updates/deletes require base_version")
            try:
                base = self.store.get_sop(candidate.target_sop_id, candidate.base_version)
            except NotFoundError as error:
                raise ValueError(
                    f"base version {candidate.base_version} is no longer retained; rebuild from version {current.version}"
                ) from error
            # Record the structure this change read, so promotion can rebase it safely.
            candidate = candidate.model_copy(update={"base_fingerprint": procedure_fingerprint(base.procedure)})
        decision = self._safety_decision(candidate, current)
        utility_ok = candidate.claimed_utility(self.config.utility_cost_weight) >= self.config.min_net_benefit
        status = CandidateStatus.PENDING if decision.allowed and utility_ok else CandidateStatus.SAFETY_REJECTED
        candidate = candidate.model_copy(update={"status": status})
        self.store.put_candidate(candidate)
        return candidate, decision

    def validate_candidate(self, candidate_id: str) -> tuple[SOPCandidate, SafetyDecision]:
        """Check state evidence and causal confidence while preserving promoted state."""
        candidate = self.store.get_candidate(candidate_id)
        if not self.config.enable_procedural_memory:
            decision = SafetyDecision(
                allowed=False,
                risk_score=0.0,
                risk_threshold=self.config.safety_risk_threshold,
                violations=["procedural memory disabled for this experiment condition"],
            )
            if candidate.status != CandidateStatus.PROMOTED:
                candidate = self.store.update_candidate_status(candidate_id, CandidateStatus.REJECTED)
            return candidate, decision
        current = self.store.get_sop(candidate.target_sop_id) if candidate.target_sop_id else None
        safety = self._safety_decision(candidate, current)
        if candidate.status == CandidateStatus.PROMOTED:
            return candidate, safety
        utility_ok = candidate.claimed_utility(self.config.utility_cost_weight) >= self.config.min_net_benefit
        if not safety.allowed or not utility_ok:
            status = CandidateStatus.SAFETY_REJECTED
        elif not candidate.state_verified or (
            self.config.require_causal_gate
            and candidate.causal_confidence < self.config.min_causal_confidence
        ):
            status = CandidateStatus.NEEDS_EVIDENCE
        else:
            status = CandidateStatus.VALIDATED
        return self.store.update_candidate_status(candidate_id, status), safety

    def record_reproduction(self, trial: ReproductionTrial) -> ReproductionTrial:
        """Record validation; the first qualified success attempts an SOP write."""
        candidate = self.store.get_candidate(trial.candidate_id)
        self.store.get_workspace(trial.workspace_id)
        self.store.add_trial(trial)
        if not self.config.enable_procedural_memory:
            return trial
        if (
            candidate.status not in {CandidateStatus.PROMOTED, CandidateStatus.SAFETY_REJECTED}
            and trial.success
            and trial.state_verified
            and (trial.causal_supported or not self.config.require_causal_gate)
        ):
            # The first qualified validation trial is sufficient. Promotion still
            # evaluates every gate, including non-negative mean net benefit, before writing.
            self.promote_candidate(trial.candidate_id)
        return trial

    def promote_candidate(self, candidate_id: str) -> PromotionDecision:
        """Recompute promotion gates and apply create/update/delete idempotently."""
        candidate = self.store.get_candidate(candidate_id)
        if not self.config.enable_procedural_memory:
            return PromotionDecision(
                promoted=False,
                candidate_id=candidate_id,
                gates={},
                reasons=["procedural memory disabled for this experiment condition"],
            )
        if candidate.status == CandidateStatus.PROMOTED:
            sop_id = candidate.target_sop_id or candidate.candidate_id
            return PromotionDecision(
                promoted=True,
                candidate_id=candidate_id,
                sop=self.store.get_sop(sop_id),
                gates={name: True for name in self.PROMOTION_GATES},
                reasons=["candidate already promoted"],
            )
        current = self.store.get_sop(candidate.target_sop_id) if candidate.target_sop_id else None
        safety = self._safety_decision(candidate, current)
        trials = self.store.list_trials(candidate_id)
        # A qualified trial needs success, state verification, and external causal support.
        qualified = [
            trial
            for trial in trials
            if trial.success
            and trial.state_verified
            and (trial.causal_supported or not self.config.require_causal_gate)
        ]
        task_families = {trial.task_family for trial in qualified}
        workspaces = {trial.workspace_id for trial in qualified}
        environments = {trial.environment_fingerprint for trial in qualified}
        # Eq. 14: average over all recorded trials, failed ones included; no trials -> -inf.
        benefits = [trial.net_benefit(self.config.utility_cost_weight) for trial in trials]
        mean_benefit = fmean(benefits) if benefits else float("-inf")
        # Default thresholds are one, so the first qualified trial satisfies reproduction.
        reproduction_ok = (
            len(qualified) >= self.config.min_reproductions
            and len(task_families) >= self.config.min_distinct_task_families
            and len(workspaces) >= self.config.min_reproductions
            and len(environments) >= self.config.min_reproductions
        )
        # Eq. 9. The claimed utility (Eq. 12) is only a precheck and is not a promotion gate.
        gates = {
            "safety": safety.allowed,
            "state_verified": candidate.state_verified and all(trial.state_verified for trial in qualified),
            "causal_attribution": (
                not self.config.require_causal_gate
                or (
                    candidate.causal_confidence >= self.config.min_causal_confidence
                    and all(trial.causal_supported for trial in qualified)
                )
            ),
            "reproduction": reproduction_ok,
            "net_benefit": mean_benefit >= self.config.min_net_benefit,
        }
        reasons = [name for name, passed in gates.items() if not passed]
        if reasons:
            if candidate.status != CandidateStatus.SAFETY_REJECTED:
                self.store.update_candidate_status(candidate_id, CandidateStatus.NEEDS_EVIDENCE)
            return PromotionDecision(
                promoted=False,
                candidate_id=candidate_id,
                gates=gates,
                reasons=[f"failed gate: {reason}" for reason in reasons],
            )
        if candidate.operation == ProposalOperation.DELETE:
            assert candidate.target_sop_id is not None and candidate.base_version is not None
            try:
                self._retire(candidate.target_sop_id, candidate.base_version, candidate.base_fingerprint, candidate_id)
            except ConcurrencyError:
                return self._superseded(candidate_id, gates)
            return PromotionDecision(
                promoted=True,
                candidate_id=candidate_id,
                gates=gates,
                reasons=["SOP safely deactivated; current and previous versions retained"],
            )
        assert candidate.procedure is not None and candidate.metadata is not None
        # A create candidate owns a deterministic SOP id. This makes retries and
        # concurrent promotion attempts converge on the same versioned record.
        sop_id = candidate.target_sop_id or candidate.candidate_id
        # Validation summary score (Eq. 15); stored as metadata, not a promotion gate.
        validation_score = min(
            1.0,
            0.4 * candidate.causal_confidence
            + 0.3 * (len(qualified) / max(self.config.min_reproductions, len(qualified)))
            + 0.3 * (1.0 - safety.risk_score),
        )

        def build(head: SOPVersion | None) -> SOPVersion:
            metadata = candidate.metadata
            rebased = head is not None and head.version != candidate.base_version
            if rebased:
                # Only warnings changed since the base: keep lessons added in between.
                merged, _ = _merge_warnings(metadata.warnings, head.metadata.warnings)
                metadata = metadata.model_copy(update={"warnings": merged})
            return SOPVersion(
                sop_id=sop_id,
                version=head.version + 1 if head else 1,
                procedure=candidate.procedure,
                metadata=metadata,
                success_count=len(qualified),
                failure_count=len(trials) - len(qualified),
                validation_score=validation_score,
                mean_net_benefit=mean_benefit,
                change=SOPChange(
                    kind="structure" if head else "create",
                    base_version=candidate.base_version,
                    base_fingerprint=candidate.base_fingerprint,
                    rebased_onto=head.version if rebased else None,
                ),
                created_at=utc_now(),
            )

        try:
            sop = self._rebased_commit(
                sop_id, candidate.base_version, candidate.base_fingerprint, build, candidate_id=candidate_id
            )
        except ConcurrencyError:
            return self._superseded(candidate_id, gates)
        return PromotionDecision(
            promoted=True,
            candidate_id=candidate_id,
            sop=sop,
            gates=gates,
            reasons=["all promotion gates passed"],
        )

    def _superseded(self, candidate_id: str, gates: dict[str, bool]) -> PromotionDecision:
        """A conflicting head (Eq. 16) leaves the active revision unchanged."""
        self.store.update_candidate_status(candidate_id, CandidateStatus.SUPERSEDED)
        return PromotionDecision(
            promoted=False,
            candidate_id=candidate_id,
            gates=gates,
            reasons=["SOP structure changed after the candidate was built; candidate superseded"],
        )

    def _rebasable_head(self, sop_id: str, base_version: int, base_fingerprint: str | None) -> SOPVersion:
        """Return the head if a change built on ``base_version`` can apply to it.

        Every change reads the base's steps and dependencies. If the head still has
        that structure, everything committed since the base only appended warnings,
        so the change does not conflict and is replayed on the head.
        """
        head = self.store.get_sop(sop_id)
        if head.version != base_version and procedure_fingerprint(head.procedure) != base_fingerprint:
            raise ConcurrencyError(
                f"SOP {sop_id!r} structure changed between version {base_version} and {head.version}"
            )
        return head

    def _rebased_commit(
        self,
        sop_id: str,
        base_version: int | None,
        base_fingerprint: str | None,
        build: Callable[[SOPVersion | None], SOPVersion | None],
        *,
        candidate_id: str | None = None,
    ) -> SOPVersion | None:
        """Commit ``build(head)`` with CAS, rebasing onto newer non-conflicting heads.

        Returns the committed version, or None when ``build`` has nothing to add.
        Raises ConcurrencyError on a structural conflict or persistent contention.
        """
        for _ in range(self.MAX_REBASE_ATTEMPTS):
            head = None if base_version is None else self._rebasable_head(sop_id, base_version, base_fingerprint)
            revision = build(head)
            if revision is None:
                return None
            try:
                self.store.commit_sop(
                    revision, expected_version=head.version if head else None, candidate_id=candidate_id
                )
                return revision
            except ConcurrencyError:
                # Another writer moved the head between the read and the commit; re-check it.
                continue
        raise ConcurrencyError(f"SOP {sop_id!r} kept changing; change not applied")

    def _retire(self, sop_id: str, base_version: int, base_fingerprint: str | None, candidate_id: str) -> None:
        for _ in range(self.MAX_REBASE_ATTEMPTS):
            head = self._rebasable_head(sop_id, base_version, base_fingerprint)
            try:
                self.store.deactivate_sop(sop_id, head.version, candidate_id)
                return
            except ConcurrencyError:
                continue
        raise ConcurrencyError(f"SOP {sop_id!r} kept changing; retirement not applied")

    def attach_warnings(self, base: SOPVersion, warnings: list[ConditionalWarning]) -> SOPVersion:
        """Attach supported conditional lessons learned while executing ``base`` (Alg. 1, line 18).

        Warnings change neither steps nor dependencies, so they need no reproduction
        trial, but each must cite supporting evidence. If other warnings were added
        after ``base``, the lessons are merged into the newest version; if the
        structure changed, ConcurrencyError is raised because the lessons were
        learned on a different task graph. Returns the resulting head.
        """
        if not self.config.enable_procedural_memory:
            raise ValueError("procedural memory disabled for this experiment condition")
        if any(not warning.evidence for warning in warnings):
            raise ValueError("conditional warnings require supporting evidence")
        base_fingerprint = procedure_fingerprint(base.procedure)

        def build(head: SOPVersion | None) -> SOPVersion | None:
            assert head is not None
            if not head.active:
                raise ValueError("warnings can only be attached to an active SOP")
            merged, added = _merge_warnings(head.metadata.warnings, warnings)
            if not added:
                return None
            return head.model_copy(update={
                "version": head.version + 1,
                "metadata": head.metadata.model_copy(update={"warnings": merged}),
                "change": SOPChange(
                    kind="warnings",
                    base_version=base.version,
                    base_fingerprint=base_fingerprint,
                    rebased_onto=head.version if head.version != base.version else None,
                    added_warnings=len(added),
                ),
                "created_at": utc_now(),
            })

        committed = self._rebased_commit(base.sop_id, base.version, base_fingerprint, build)
        return committed or self.store.get_sop(base.sop_id)

    def _safety_decision(
        self,
        candidate: SOPCandidate,
        current: SOPVersion | None,
    ) -> SafetyDecision:
        """Unified safety entry point; disabling it is only for isolated ablations."""
        if self.config.enable_safety_gate:
            if current is None and candidate.metadata and candidate.metadata.variant_of:
                current = self.store.get_sop(candidate.metadata.variant_of)
            return self.safety.validate(candidate, current)
        return SafetyDecision(
            allowed=True,
            risk_score=0.0,
            violations=[],
            mandatory_steps_preserved=True,
        )

    # ---------- Retrieval and real execution feedback ----------
    def retrieve_sops(
        self, query: str, limit: int = 5, *, context_facts: set[str] | None = None,
    ) -> list[RetrievalResult]:
        """Return the top-K active SOPs compatible with the verified context facts."""
        if not self.config.enable_procedural_memory:
            return []
        sops, _ = self.store.list_active_sops(limit=1_000, offset=0)
        return self.retriever.rank(query, sops, limit, context_facts=context_facts)

    def retrieve_trial_candidates(
        self,
        query: str,
        limit: int = 1,
        *,
        context_facts: set[str] | None = None,
        task_family: str | None = None,
    ) -> list[RetrievalResult]:
        """Rank unpromoted create candidates that recorded trials could still promote.

        Candidates stay outside the active pool (Appendix B.5); this only selects
        which one a host may run as a reproduction trial. Candidates that fail a
        candidate-level gate (unverified state, low causal confidence) are skipped
        because no trial can promote them.
        """
        if not self.config.enable_procedural_memory:
            return []
        pool: list[SOPVersion] = []
        for status in (CandidateStatus.VALIDATED, CandidateStatus.NEEDS_EVIDENCE):
            candidates, _ = self.store.list_candidates(status, limit=1_000, offset=0)
            pool.extend(
                SOPVersion(sop_id=item.candidate_id, version=1, procedure=item.procedure, metadata=item.metadata)
                for item in candidates
                if item.operation == ProposalOperation.CREATE
                and item.procedure is not None
                and item.metadata is not None
                and item.state_verified
                and (
                    not self.config.require_causal_gate
                    or item.causal_confidence >= self.config.min_causal_confidence
                )
                and (task_family is None or item.metadata.task_family == task_family)
            )
        return self.retriever.rank(query, pool, limit, context_facts=context_facts)

    def record_sop_outcome(self, sop_id: str, success: bool) -> SOPVersion:
        """Update the active SOP's usage and success statistics."""
        return self.store.record_sop_outcome(sop_id, success)
