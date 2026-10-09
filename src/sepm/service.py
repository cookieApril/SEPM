"""Business orchestration layer and recommended facade for SEPM.

The service connects storage, divergence detection, evidence resolution, safety,
SOP promotion, and retrieval into one workflow. Framework adapters, the CLI,
benchmarks, tests, and examples reuse it instead of duplicating rules.

The publication path is propose_sop -> validate_candidate ->
record_reproduction. The first qualified successful trial immediately calls
promote_candidate, which rechecks every gate and commits through storage CAS.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from .config import MemoryConfig
from .divergence import DivergenceDetector, GoalJudge
from .embedding import Embedder, HashingEmbedder
from .evidence import EvidenceResolver
from .models import (
    AgentProfile,
    BlackboardEntry,
    CandidateStatus,
    DivergenceReport,
    Evidence,
    PromotionDecision,
    ProposalOperation,
    ReproductionTrial,
    Resolution,
    RetrievalResult,
    SafetyDecision,
    SOPCandidate,
    SOPVersion,
    TaskPlan,
    Workspace,
    utc_now,
)
from .retrieval import AdaptiveWeightRouter, SOPRetriever
from .safety import SafetyValidator
from .storage import NotFoundError, SQLiteStore


def _canonical(value: Any) -> str:
    """Serialize state values stably so key order does not affect conflict detection."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


class SEPMService:
    """Shared facade with injectable dependencies for tests and experiments."""

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
        self.store = SQLiteStore(self.config.database_path)
        self.embedder = embedder or HashingEmbedder()
        self.detector = DivergenceDetector(self.embedder, goal_judge, self.config)
        self.resolver = EvidenceResolver(self.config)
        self.safety = SafetyValidator(self.config)
        self.router = AdaptiveWeightRouter(self.embedder, self.config)
        self.retriever = SOPRetriever(self.embedder, self.router, self.config)

    # ---------- Private memory and shared task state ----------
    def register_agent(self, profile: AgentProfile) -> AgentProfile:
        self.store.upsert_agent(profile)
        return profile

    def create_workspace(self, workspace: Workspace) -> Workspace:
        self.store.put_workspace(workspace, overwrite=False)
        return self.store.get_workspace(workspace.workspace_id)

    def append_blackboard(self, entry: BlackboardEntry) -> BlackboardEntry:
        """Append an event after validating agent/workspace without changing the goal."""
        self.store.get_agent(entry.agent_id)
        workspace = self.store.get_workspace(entry.workspace_id)
        self.store.append_entry(entry, self.config.max_blackboard_entries)
        if entry.goal and entry.goal != workspace.main_goal:
            # Detection is explicit; appending never silently mutates canonical state.
            pass
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
        conflicts = self._state_conflicts(workspace_id, focus_agent=agent_id)
        return self.detector.inspect(workspace, latest, conflicts)

    def resolve_state(
        self,
        workspace_id: str,
        state_key: str,
        agent_id: str | None = None,
    ) -> Resolution:
        """Collect evidence for one key; bare claims become low-confidence inference."""
        if not self.config.share_blackboard_across_agents and agent_id is None:
            raise ValueError("agent_id is required when shared blackboard is disabled")
        entries = self.store.list_state_entries(workspace_id, state_key, agent_id=agent_id)
        evidence: list[Evidence] = []
        for entry in entries:
            if entry.evidence:
                evidence.extend(entry.evidence)
            elif entry.state_value is not None:
                # An unsupported state claim is only an agent inference.
                from .models import EvidenceTier

                evidence.append(
                    Evidence(
                        tier=EvidenceTier.AGENT_INFERENCE,
                        source=f"agent:{entry.agent_id}",
                        value=entry.state_value,
                        observed_at=entry.created_at,
                        confidence=0.5,
                    )
                )
        return self.resolver.resolve_world_state(state_key, evidence)

    def _state_conflicts(self, workspace_id: str, focus_agent: str | None = None) -> list[str]:
        """Find state keys claimed as different normalized values by multiple agents."""
        local_agent = focus_agent if not self.config.share_blackboard_across_agents else None
        entries = self.store.list_state_entries(workspace_id, agent_id=local_agent)
        values: dict[str, dict[str, str]] = defaultdict(dict)
        for entry in entries:
            if entry.state_key is not None:
                # Storage returns newest first. Compare current beliefs, not all historical values.
                values[entry.state_key].setdefault(entry.agent_id, _canonical(entry.state_value))
        conflicts: list[str] = []
        for key, per_agent in values.items():
            if focus_agent and focus_agent not in per_agent:
                continue
            union = set(per_agent.values())
            if len(union) > 1:
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
            # evaluates every gate, including positive net benefit, before writing.
            self.promote_candidate(trial.candidate_id)
        return trial

    def promote_candidate(self, candidate_id: str) -> PromotionDecision:
        """Recompute promotion gates and apply create/update/delete idempotently."""
        candidate = self.store.get_candidate(candidate_id)
        if candidate.status == CandidateStatus.PROMOTED:
            sop_id = candidate.target_sop_id or candidate.candidate_id
            return PromotionDecision(
                promoted=True,
                candidate_id=candidate_id,
                sop=self.store.get_sop(sop_id),
                gates={
                    "safety": True,
                    "state_verified": True,
                    "causal_attribution": True,
                    "cross_task_reproduction": True,
                    "net_benefit": True,
                },
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
        # Failed evaluations are evidence too; conditioning utility on success biases promotion.
        benefits = [
            trial.candidate_reward - trial.baseline_reward
            - self.config.utility_cost_weight * trial.cost
            for trial in trials
        ]
        mean_benefit = fmean(benefits) if benefits else float("-inf")
        # Default thresholds are one, so the first qualified trial satisfies legacy gates.
        reproduction_ok = (
            len(qualified) >= self.config.min_reproductions
            and len(task_families) >= self.config.min_distinct_task_families
            and len(workspaces) >= self.config.min_reproductions
            and len(environments) >= self.config.min_reproductions
        )
        gates = {
            "procedural_memory_enabled": self.config.enable_procedural_memory,
            "utility": candidate.claimed_utility(self.config.utility_cost_weight) >= self.config.min_net_benefit,
            "safety_validation": safety.allowed,
            "state_verified": candidate.state_verified and all(trial.state_verified for trial in qualified),
            "causal_attribution": (
                not self.config.require_causal_gate
                or (
                    candidate.causal_confidence >= self.config.min_causal_confidence
                    and all(trial.causal_supported for trial in qualified)
                )
            ),
            "cross_task_reproduction": reproduction_ok,
            "net_benefit": mean_benefit > self.config.min_net_benefit,
        }
        reasons = [name for name, passed in gates.items() if not passed]
        if reasons:
            self.store.update_candidate_status(candidate_id, CandidateStatus.NEEDS_EVIDENCE)
            return PromotionDecision(
                promoted=False,
                candidate_id=candidate_id,
                gates=gates,
                reasons=[f"failed gate: {reason}" for reason in reasons],
            )
        if candidate.operation == ProposalOperation.DELETE:
            assert current is not None and candidate.base_version is not None
            self.store.deactivate_sop(current.sop_id, candidate.base_version, candidate_id)
            return PromotionDecision(
                promoted=True,
                candidate_id=candidate_id,
                gates=gates,
                reasons=["SOP safely deactivated; version history retained"],
            )
        assert candidate.procedure is not None and candidate.metadata is not None
        # A create candidate owns a deterministic SOP id. This makes retries and
        # concurrent promotion attempts converge on the same versioned record.
        sop_id = candidate.target_sop_id or candidate.candidate_id
        next_version = (current.version + 1) if current else 1
        # Validation score combines causal confidence, trial coverage, and safety risk.
        validation_score = min(
            1.0,
            0.4 * candidate.causal_confidence
            + 0.3 * (len(qualified) / max(self.config.min_reproductions, len(qualified)))
            + 0.3 * (1.0 - safety.risk_score),
        )
        sop = SOPVersion(
            sop_id=sop_id,
            version=next_version,
            procedure=candidate.procedure,
            metadata=candidate.metadata,
            success_count=len(qualified),
            failure_count=len(trials) - len(qualified),
            validation_score=validation_score,
            mean_net_benefit=mean_benefit,
            created_at=utc_now(),
        )
        self.store.commit_sop(sop, expected_version=candidate.base_version, candidate_id=candidate_id)
        return PromotionDecision(
            promoted=True,
            candidate_id=candidate_id,
            sop=sop,
            gates=gates,
            reasons=["all promotion gates passed"],
        )

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
        self, query: str, limit: int = 5, query_plan: TaskPlan | None = None,
        *, context_facts: set[str] | None = None,
    ) -> list[RetrievalResult]:
        if not self.config.enable_procedural_memory:
            return []
        sops, _ = self.store.list_active_sops(limit=1_000, offset=0)
        return self.retriever.rank(query, sops, limit, query_plan, context_facts=context_facts)

    def record_sop_outcome(self, sop_id: str, success: bool, preferred_feature: str | None = None, query: str = "") -> SOPVersion:
        """Update SOP usage/success statistics and optionally train the retrieval router."""
        updated = self.store.record_sop_outcome(sop_id, success)
        if preferred_feature:
            self.router.learn(query, preferred_feature)
        return updated
