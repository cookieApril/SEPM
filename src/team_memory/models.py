"""定义 Team Memory 跨层传递的严格领域模型。

所有外部输入先经过这些 Pydantic 模型验证：未知字段被拒绝、字符串去除首尾空白、赋值
时继续校验。模型按“共享任务状态 → 证据与偏差 → SOP 生命周期 → 检索结果”排列，
是 Python 服务、适配器和 benchmark 共享的数据契约。
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    """统一生成带 UTC 时区的时间，避免本地时区混入排序和审计。"""
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    """所有领域对象的严格基类，禁止调用方悄悄传入未声明字段。"""
    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)


# ---------- 共享任务状态与偏差枚举 ----------

class BlackboardKind(str, Enum):
    """黑板事件类型；只存显式事实/动作，不存隐藏推理链。"""
    TASK = "task"
    OBSERVATION = "observation"
    ACTION = "action"
    RESULT = "result"
    ERROR = "error"
    STATE_CLAIM = "state_claim"
    PLAN = "plan"


class EvidenceTier(IntEnum):
    """数值越大权威性越高，EvidenceResolver 依此选择最高等级。"""
    AGENT_INFERENCE = 1
    VERIFIED_ARTIFACT = 2
    TOOL_OBSERVATION = 3
    AUTHORITATIVE_STATE = 4


class ConflictType(str, Enum):
    """系统可以结构化报告的四类冲突。"""
    REASONING = "reasoning"
    WORLD_STATE = "world_state"
    GOAL = "goal"
    PLAN = "plan"


class ProposalOperation(str, Enum):
    """候选对 SOP head 的期望操作。"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"


class CandidateStatus(str, Enum):
    """候选队列状态机；PROMOTED 是成功终态，SAFETY_REJECTED 是安全拒绝态。"""
    PENDING = "pending"
    SAFETY_REJECTED = "safety_rejected"
    NEEDS_EVIDENCE = "needs_evidence"
    VALIDATED = "validated"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


# ---------- Agent、计划、工作区和证据 ----------

class AgentProfile(StrictModel):
    """Agent 的私有角色信息；permanent_facts 不进入共享黑板。"""
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
    """动作节点和依赖边组成的计划图。"""
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


class Workspace(StrictModel):
    """一个协作任务的规范目标与规范计划快照。"""
    workspace_id: str = Field(default_factory=lambda: str(uuid4()))
    main_goal: str = Field(min_length=1, max_length=10_000)
    goal_version: int = Field(default=1, ge=1)
    plan: TaskPlan = Field(default_factory=TaskPlan)
    plan_version: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=utc_now)


class Evidence(StrictModel):
    """带来源、权威等级、观测时间和置信度的可审计证据。"""
    evidence_id: str = Field(default_factory=lambda: str(uuid4()))
    tier: EvidenceTier
    source: str = Field(min_length=1, max_length=512)
    value: Any
    observed_at: datetime = Field(default_factory=utc_now)
    verifier: str | None = Field(default=None, max_length=512)
    artifact_uri: str | None = Field(default=None, max_length=2_000)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class BlackboardEntry(StrictModel):
    """追加式共享事件；按 kind 选择填写任务、观测、动作、结果或状态字段。"""
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


# ---------- 偏差检测与状态裁决输出 ----------

class DivergenceReport(StrictModel):
    """三类偏差的量化结果以及是否需要对齐。"""
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


class Resolution(StrictModel):
    """证据裁决结果；未解决时 next_action 明确指出下一步取证动作。"""
    conflict_type: ConflictType
    state_key: str | None = None
    resolved: bool
    value: Any = None
    winning_evidence: Evidence | None = None
    considered_evidence: list[Evidence] = Field(default_factory=list)
    next_action: str


# ---------- SOP 候选、验证 trial 与不可变版本 ----------

class ProcedureStep(StrictModel):
    """SOP 的原子步骤，包含前后置条件和显式安全属性。"""
    step_id: str = Field(default_factory=lambda: str(uuid4()))
    instruction: str = Field(min_length=1, max_length=4_000)
    action_type: str = Field(default="generic", min_length=1, max_length=128)
    requires_confirmation: bool = False
    preconditions: list[str] = Field(default_factory=list, max_length=50)
    postconditions: list[str] = Field(default_factory=list, max_length=50)
    safety_critical: bool = False


class ProcedureGraph(StrictModel):
    """SOP 步骤及其依赖边；边端点必须引用已有 step_id。"""
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


class SOPMetadata(StrictModel):
    """影响适用范围、检索和安全策略的 SOP 描述信息。"""
    title: str = Field(min_length=1, max_length=512)
    description: str = Field(default="", max_length=8_000)
    task_family: str = Field(min_length=1, max_length=256)
    applicability: list[str] = Field(default_factory=list, max_length=100)
    exclusions: list[str] = Field(default_factory=list, max_length=100)
    tags: list[str] = Field(default_factory=list, max_length=100)
    safety_class: str = Field(default="normal", pattern="^(normal|sensitive|critical)$")
    variant_of: str | None = Field(default=None, max_length=128)
    context_conditions: list[str] = Field(default_factory=list, max_length=100)
    warnings: list[str] = Field(default_factory=list, max_length=100)


class SOPCandidate(StrictModel):
    """尚未发布的 create/update/delete 提案及其证据摘要。"""
    candidate_id: str = Field(default_factory=lambda: str(uuid4()))
    operation: ProposalOperation
    target_sop_id: str | None = None
    base_version: int | None = Field(default=None, ge=1)
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
    def validate_operation(self) -> "SOPCandidate":
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
        """Return Success - lambda * Cost for the candidate's declared update value."""
        return self.claimed_benefit - cost_weight * self.claimed_cost

    @field_validator("procedure")
    @classmethod
    def create_update_need_procedure(cls, value: ProcedureGraph | None, info: Any) -> ProcedureGraph | None:
        if info.data.get("operation") in {ProposalOperation.CREATE, ProposalOperation.UPDATE} and value is None:
            raise ValueError("create/update proposals require a procedure")
        return value


class SafetyDecision(StrictModel):
    """确定性安全门输出。"""
    allowed: bool
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_threshold: float = Field(default=1.0, ge=0.0, le=1.0)
    violations: list[str] = Field(default_factory=list)
    mandatory_steps_preserved: bool = True


class ReproductionTrial(StrictModel):
    """候选与 baseline 的一次隔离验证；首个合格成功 trial 即可触发发布。"""
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

    @property
    def net_benefit(self) -> float:
        """候选相对 baseline 的收益扣除执行成本。"""
        return self.candidate_reward - self.baseline_reward - self.cost


class SOPVersion(StrictModel):
    """已发布且内容不可变的 SOP 版本；运行统计允许原位更新。"""
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
    created_at: datetime = Field(default_factory=utc_now)


# ---------- 检索与晋升 API 输出 ----------

class RetrievalResult(StrictModel):
    """单条检索结果，保留特征与权重便于解释排序。"""
    sop: SOPVersion
    score: float = Field(ge=0.0, le=1.0)
    features: dict[str, float]
    weights: dict[str, float]
    explanation: str


class PromotionDecision(StrictModel):
    """五门逐项结果；失败时 sop 为空、reasons 指出未通过门。"""
    promoted: bool
    candidate_id: str
    sop: SOPVersion | None = None
    gates: dict[str, bool]
    reasons: list[str]
