"""Strict domain models shared across SEPM layers.

All external input passes through these Pydantic models. Unknown fields are
rejected, strings are stripped, and assignment remains validated. Models are
ordered from shared task state through evidence/divergence and the SOP lifecycle
to retrieval results. They form the data contract shared by services, adapters,
and benchmarks.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    """Generate UTC-aware timestamps so local time zones do not affect ordering."""
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    """Strict base class that rejects undeclared fields on every domain object."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)


# ---------- Shared task state and divergence enumerations ----------

class BlackboardKind(str, Enum):
    """Blackboard event kinds; store explicit facts/actions, not hidden reasoning."""
    TASK = "task"
    OBSERVATION = "observation"
    ACTION = "action"
    RESULT = "result"
    ERROR = "error"
    STATE_CLAIM = "state_claim"
    PLAN = "plan"


class EvidenceTier(IntEnum):
    """Higher values mean greater authority and determine resolver priority."""
    AGENT_INFERENCE = 1
    VERIFIED_ARTIFACT = 2
    TOOL_OBSERVATION = 3
    AUTHORITATIVE_STATE = 4


class ConflictType(str, Enum):
    """The four conflict classes that the system can report structurally."""
    REASONING = "reasoning"
    WORLD_STATE = "world_state"
    GOAL = "goal"
    PLAN = "plan"


class ProposalOperation(str, Enum):
    """Operation requested by a candidate against an SOP head."""
    CREATE = "create"
    UPDATE = "update"
    # Retirement: deactivates the SOP head while keeping its retained versions.
    DELETE = "delete"


class CandidateStatus(str, Enum):
    """Candidate queue states; PROMOTED succeeds and SAFETY_REJECTED is terminal."""
    PENDING = "pending"
    SAFETY_REJECTED = "safety_rejected"
    NEEDS_EVIDENCE = "needs_evidence"
    VALIDATED = "validated"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


# ---------- Agents, plans, workspaces, and evidence ----------

class AgentProfile(StrictModel):
    """Private agent role data."""
    agent_id: str = Field(min_length=1, max_length=128)
    role: str = Field(min_length=1, max_length=512)
    permanent_facts: dict[str, Any] = Field(default_factory=dict)
    current_task: str | None = Field(default=None, max_length=4_000)


class PlanNode(StrictModel):
    node_id: str = Field(min_length=1, max_length=128)
    action: str = Field(min_length=1, max_length=1_000)


class PlanEdge(StrictModel):
    source: str = Field(min_length=1, max_length=128)
    target: str = Field(min_length=1, max_length=128)


def _validate_dependencies(ids: set[str], edges: list[PlanEdge]) -> None:
    pairs = {(edge.source, edge.target) for edge in edges}
    if len(pairs) != len(edges):
        raise ValueError("dependency graph cannot contain duplicate edges")
    outgoing = {node: [] for node in ids}
    incoming = {node: 0 for node in ids}
    for source, target in pairs:
        outgoing[source].append(target)
        incoming[target] += 1
    ready = [node for node, count in incoming.items() if count == 0]
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for target in outgoing[node]:
            incoming[target] -= 1
            if incoming[target] == 0:
                ready.append(target)
    if visited != len(ids):
        raise ValueError("dependency graph must be acyclic")


class TaskPlan(StrictModel):
    """Plan graph composed of action nodes and dependency edges."""
    nodes: list[PlanNode] = Field(default_factory=list, max_length=500)
    edges: list[PlanEdge] = Field(default_factory=list, max_length=2_000)

    @field_validator("nodes")
    @classmethod
    def unique_node_ids(cls, value: list[PlanNode]) -> list[PlanNode]:
        ids = [node.node_id for node in value]
        if len(ids) != len(set(ids)):
            raise ValueError("Plan graph node_id values must be unique")
        return value

    @field_validator("edges")
    @classmethod
    def validate_edges(cls, value: list[PlanEdge], info: Any) -> list[PlanEdge]:
        if any(edge.source == edge.target for edge in value):
            raise ValueError("Plan graph cannot contain self-loops")
        ids = {node.node_id for node in info.data.get("nodes", [])}
        missing = {endpoint for edge in value for endpoint in (edge.source, edge.target)} - ids
        if missing:
            raise ValueError(f"Plan edges reference unknown nodes: {sorted(missing)}")
        _validate_dependencies(ids, value)
        return value


class AgentAssignment(StrictModel):
    """Manager binding of one agent to its reference goal and task-graph nodes."""
    goal: str | None = Field(default=None, max_length=10_000)
    node_ids: list[str] = Field(default_factory=list, max_length=500)


def assigned_subgraph(plan: TaskPlan, node_ids: list[str]) -> TaskPlan:
    """Return an agent's assigned nodes plus their direct prerequisites from other agents."""
    assigned = set(node_ids)
    keep = assigned | {edge.source for edge in plan.edges if edge.target in assigned}
    return TaskPlan(
        nodes=[node for node in plan.nodes if node.node_id in keep],
        edges=[edge for edge in plan.edges if edge.source in keep and edge.target in keep],
    )


class Workspace(StrictModel):
    """Canonical goal, task graph, and agent assignments for one collaborative task."""
    workspace_id: str = Field(default_factory=lambda: str(uuid4()))
    main_goal: str = Field(min_length=1, max_length=10_000)
    goal_version: int = Field(default=1, ge=1)
    plan: TaskPlan = Field(default_factory=TaskPlan)
    plan_version: int = Field(default=1, ge=1)
    assignments: dict[str, AgentAssignment] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def assignments_reference_plan(self) -> Workspace:
        known = {node.node_id for node in self.plan.nodes}
        for agent_id, assignment in self.assignments.items():
            missing = set(assignment.node_ids) - known
            if missing:
                raise ValueError(f"assignment for {agent_id!r} references unknown nodes: {sorted(missing)}")
        return self

    def reference_goal(self, agent_id: str) -> str:
        """Assigned goal g*_{tau,a}; the global objective when the agent has no assignment."""
        assignment = self.assignments.get(agent_id)
        return assignment.goal if assignment and assignment.goal else self.main_goal

    def reference_plan(self, agent_id: str) -> TaskPlan:
        """Assigned subgraph G_{tau,a}; the full task graph when no nodes are assigned."""
        assignment = self.assignments.get(agent_id)
        if assignment is None or not assignment.node_ids:
            return self.plan
        return assigned_subgraph(self.plan, assignment.node_ids)


class Evidence(StrictModel):
    """Auditable evidence with source, authority, observation time, and confidence."""
    evidence_id: str = Field(default_factory=lambda: str(uuid4()))
    tier: EvidenceTier
    source: str = Field(min_length=1, max_length=512)
    value: Any
    observed_at: datetime = Field(default_factory=utc_now)
    verifier: str | None = Field(default=None, max_length=512)
    artifact_uri: str | None = Field(default=None, max_length=2_000)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class BlackboardEntry(StrictModel):
    """Append-only shared event with fields selected according to its kind."""
    entry_id: str = Field(default_factory=lambda: str(uuid4()))
    workspace_id: str = Field(min_length=1, max_length=128)
    agent_id: str = Field(min_length=1, max_length=128)
    kind: BlackboardKind
    task: str | None = Field(default=None, max_length=4_000)
    observation: str | None = Field(default=None, max_length=8_000)
    action: str | None = Field(default=None, max_length=4_000)
    result: str | None = Field(default=None, max_length=8_000)
    error: str | None = Field(default=None, max_length=8_000)
    state_key: str | None = Field(default=None, max_length=512)
    state_value: Any = None
    goal: str | None = Field(default=None, max_length=10_000)
    plan: TaskPlan | None = None
    evidence: list[Evidence] = Field(default_factory=list, max_length=50)
    created_at: datetime = Field(default_factory=utc_now)


# ---------- Divergence detection and state-resolution outputs ----------

class DivergenceReport(StrictModel):
    """Quantified results for three divergence classes and alignment need."""
    agent_id: str
    goal_observed: bool = True
    plan_observed: bool = True
    goal_divergence: float = Field(ge=0.0, le=1.0)
    goal_embedding_component: float = Field(ge=0.0, le=1.0)
    goal_judge_component: float = Field(ge=0.0, le=1.0)
    plan_divergence: float = Field(ge=0.0, le=1.0)
    conflicting_state_keys: list[str] = Field(default_factory=list)
    requires_alignment: bool
    reasons: list[str] = Field(default_factory=list)
    alignment_stage: str = Field(default="detect", pattern="^(detect|classify|verify|resolve|realign|none)$")
    recommended_action: str = Field(default="", max_length=2_000)


class AlignmentResult(StrictModel):
    """Blackboard-derived alignment of one agent and the corrections to propagate."""
    report: DivergenceReport
    missing_prerequisites: list[str] = Field(default_factory=list)
    off_plan_actions: list[str] = Field(default_factory=list)
    resolved_state: dict[str, Any] = Field(default_factory=dict)
    notices: list[str] = Field(default_factory=list)


class Resolution(StrictModel):
    """Evidence resolution; next_action specifies evidence collection if unresolved."""
    conflict_type: ConflictType
    state_key: str | None = None
    resolved: bool
    value: Any = None
    winning_evidence: Evidence | None = None
    considered_evidence: list[Evidence] = Field(default_factory=list)
    next_action: str


# ---------- SOP candidates, validation trials, and versions ----------

class ProcedureStep(StrictModel):
    """Atomic SOP step with preconditions, postconditions, and explicit safety fields."""
    step_id: str = Field(default_factory=lambda: str(uuid4()))
    instruction: str = Field(min_length=1, max_length=4_000)
    action_type: str = Field(default="generic", min_length=1, max_length=128)
    requires_confirmation: bool = False
    preconditions: list[str] = Field(default_factory=list, max_length=50)
    postconditions: list[str] = Field(default_factory=list, max_length=50)
    safety_critical: bool = False


class ProcedureGraph(StrictModel):
    """SOP steps and dependencies; every edge endpoint must reference a step_id."""
    steps: list[ProcedureStep] = Field(min_length=1, max_length=500)
    edges: list[PlanEdge] = Field(default_factory=list, max_length=2_000)

    @field_validator("steps")
    @classmethod
    def unique_step_ids(cls, value: list[ProcedureStep]) -> list[ProcedureStep]:
        ids = [step.step_id for step in value]
        if len(ids) != len(set(ids)):
            raise ValueError("Procedure graph step_id values must be unique")
        return value

    @field_validator("edges")
    @classmethod
    def validate_graph(cls, value: list[PlanEdge], info: Any) -> list[PlanEdge]:
        steps = info.data.get("steps", [])
        ids = {step.step_id for step in steps}
        missing = {endpoint for edge in value for endpoint in (edge.source, edge.target)} - ids
        if missing:
            raise ValueError(f"Procedure edges reference unknown steps: {sorted(missing)}")
        _validate_dependencies(ids, value)
        return value

    def as_plan(self) -> TaskPlan:
        """Instantiate the procedure as a task graph whose node IDs are the step IDs."""
        return TaskPlan(
            nodes=[PlanNode(node_id=step.step_id[:128], action=step.instruction[:1_000]) for step in self.steps],
            edges=[PlanEdge(source=edge.source[:128], target=edge.target[:128]) for edge in self.edges],
        )


def procedure_fingerprint(procedure: ProcedureGraph) -> str:
    """Stable hash of steps and dependencies; warnings and statistics do not affect it."""
    canonical = json.dumps(procedure.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class ConditionalWarning(StrictModel):
    """Conditional lesson with its triggering condition and supporting evidence."""
    text: str = Field(min_length=1, max_length=2_000)
    trigger_condition: str = Field(default="", max_length=2_000)
    evidence: list[str] = Field(default_factory=list, max_length=50)


class SOPMetadata(StrictModel):
    """SOP metadata governing scope, retrieval, and safety policy."""
    title: str = Field(min_length=1, max_length=512)
    description: str = Field(default="", max_length=8_000)
    task_family: str = Field(min_length=1, max_length=256)
    applicability: list[str] = Field(default_factory=list, max_length=100)
    exclusions: list[str] = Field(default_factory=list, max_length=100)
    tags: list[str] = Field(default_factory=list, max_length=100)
    safety_class: str = Field(default="normal", pattern="^(normal|sensitive|critical)$")
    variant_of: str | None = Field(default=None, max_length=128)
    context_conditions: list[str] = Field(default_factory=list, max_length=100)
    warnings: list[ConditionalWarning] = Field(default_factory=list, max_length=100)

    @field_validator("warnings", mode="before")
    @classmethod
    def coerce_plain_warnings(cls, value: Any) -> Any:
        """Accept legacy plain-text warnings as unconditional lessons."""
        if isinstance(value, list):
            return [{"text": item} if isinstance(item, str) else item for item in value]
        return value


class SOPCandidate(StrictModel):
    """Unpublished create/update/delete proposal and its evidence summary."""
    candidate_id: str = Field(default_factory=lambda: str(uuid4()))
    operation: ProposalOperation
    target_sop_id: str | None = None
    base_version: int | None = Field(default=None, ge=1)
    # Structure the update was built from; the service fills it from base_version.
    base_fingerprint: str | None = Field(default=None, max_length=64)
    procedure: ProcedureGraph | None = None
    metadata: SOPMetadata | None = None
    source_workspace_ids: list[str] = Field(min_length=1, max_length=100)
    state_verified: bool = False
    causal_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    claimed_benefit: float = 0.0
    claimed_cost: float = Field(default=0.0, ge=0.0)
    status: CandidateStatus = CandidateStatus.PENDING
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_operation(self) -> SOPCandidate:
        if self.operation == ProposalOperation.CREATE:
            if self.target_sop_id is not None or self.base_version is not None:
                raise ValueError("create uses a new SOP id; use metadata.variant_of for specialization")
        elif self.target_sop_id is None or self.base_version is None:
            raise ValueError("update/delete require target_sop_id and base_version")
        if self.operation != ProposalOperation.DELETE:
            if self.procedure is None or self.metadata is None:
                raise ValueError("create/update require procedure and metadata")
            if self.metadata.variant_of and not self.metadata.context_conditions:
                raise ValueError("context-specific variants require context_conditions")
        return self

    def claimed_utility(self, cost_weight: float = 1.0) -> float:
        """Precheck estimate u_claim = b_claim - lambda * c_claim."""
        return self.claimed_benefit - cost_weight * self.claimed_cost


class SafetyDecision(StrictModel):
    """Output of the deterministic safety gate."""
    allowed: bool
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_threshold: float = Field(default=1.0, ge=0.0, le=1.0)
    violations: list[str] = Field(default_factory=list)
    mandatory_steps_preserved: bool = True


class ReproductionTrial(StrictModel):
    """Isolated candidate/baseline validation; the first qualified success may publish."""
    trial_id: str = Field(default_factory=lambda: str(uuid4()))
    candidate_id: str
    workspace_id: str
    task_family: str
    environment_fingerprint: str
    success: bool
    baseline_reward: float
    candidate_reward: float
    cost: float = Field(ge=0.0)
    state_verified: bool
    causal_supported: bool
    created_at: datetime = Field(default_factory=utc_now)

    def net_benefit(self, cost_weight: float = 1.0) -> float:
        """Trial utility u_t = r_candidate - r_baseline - lambda * c_t (Eq. S5)."""
        return self.candidate_reward - self.baseline_reward - cost_weight * self.cost


class SOPChange(StrictModel):
    """What a revision changed and which base it was built from.

    A structural change rewrites steps and dependencies (Pi, G); a warnings change
    only appends conditional lessons (W). Both read the base structure, so a change
    can be rebased onto a newer head only while that structure is unchanged.
    """
    kind: str = Field(pattern="^(create|structure|warnings)$")
    base_version: int | None = Field(default=None, ge=1)
    base_fingerprint: str | None = Field(default=None, max_length=64)
    rebased_onto: int | None = Field(default=None, ge=1)
    added_warnings: int = Field(default=0, ge=0)


class SOPVersion(StrictModel):
    """Published SOP content with in-place runtime statistics."""
    sop_id: str
    version: int = Field(ge=1)
    procedure: ProcedureGraph
    metadata: SOPMetadata
    success_count: int = Field(default=0, ge=0)
    failure_count: int = Field(default=0, ge=0)
    retrieval_count: int = Field(default=0, ge=0)
    validation_score: float = Field(default=0.0, ge=0.0, le=1.0)
    mean_net_benefit: float = 0.0
    active: bool = True
    change: SOPChange | None = None
    created_at: datetime = Field(default_factory=utc_now)


# ---------- Retrieval and promotion API outputs ----------

class RetrievalResult(StrictModel):
    """One retrieval result retaining features and weights for explainability."""
    sop: SOPVersion
    score: float = Field(ge=0.0, le=1.0)
    features: dict[str, float]
    weights: dict[str, float]
    explanation: str


class PromotionDecision(StrictModel):
    """Per-gate result; on failure sop is empty and reasons identify failed gates."""
    promoted: bool
    candidate_id: str
    sop: SOPVersion | None = None
    gates: dict[str, bool]
    reasons: list[str]
