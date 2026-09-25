"""Team Memory 的业务编排层和唯一推荐门面。

该服务把存储、偏差检测、证据裁决、安全门、SOP 晋升和检索连接成完整流程。Python
框架适配器、CLI、benchmark、测试与示例都复用它，避免不同入口各自实现一套规则。

关键发布路径：propose_sop → validate_candidate → record_reproduction。首个合格成功
trial 写入后会立刻调用 promote_candidate；后者重新检查全部门控并用存储层 CAS 提交。
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
    SOPCandidate,
    SOPVersion,
    SafetyDecision,
    TaskPlan,
    Workspace,
    utc_now,
)
from .retrieval import AdaptiveWeightRouter, SOPRetriever
from .safety import SafetyValidator
from .storage import NotFoundError, SQLiteStore


def _canonical(value: Any) -> str:
    """稳定序列化状态值，供冲突检测去除字典键顺序差异。"""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


class TeamMemoryService:
    """供所有入口复用的业务门面；依赖均可注入以支持测试和实验替换。"""

    def __init__(
        self,
        database_path: str | Path | None = None,
        *,
        config: MemoryConfig | None = None,
        embedder: Embedder | None = None,
        goal_judge: GoalJudge | None = None,
    ) -> None:
        """初始化同一配置下的存储、检测、裁决、安全与检索组件。"""
        self.config = config or MemoryConfig(database_path=Path(database_path or "team_memory.db"))
        if database_path is not None and Path(database_path) != self.config.database_path:
            self.config = MemoryConfig(**{**self.config.__dict__, "database_path": Path(database_path)})
        self.store = SQLiteStore(self.config.database_path)
        self.embedder = embedder or HashingEmbedder()
        self.detector = DivergenceDetector(self.embedder, goal_judge, self.config)
        self.resolver = EvidenceResolver(self.config)
        self.safety = SafetyValidator(self.config)
        self.router = AdaptiveWeightRouter(self.embedder, self.config)
        self.retriever = SOPRetriever(self.embedder, self.router, self.config)

    # ---------- 私有记忆与共享任务状态 ----------
    def register_agent(self, profile: AgentProfile) -> AgentProfile:
        self.store.upsert_agent(profile)
        return profile

    def create_workspace(self, workspace: Workspace) -> Workspace:
        self.store.put_workspace(workspace, overwrite=False)
        return self.store.get_workspace(workspace.workspace_id)

    def append_blackboard(self, entry: BlackboardEntry) -> BlackboardEntry:
        """验证 Agent/工作区存在后追加事件；不会隐式改写规范目标。"""
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
        """读取共享黑板；关闭共享消融时必须指定 Agent，防止看到其他 Agent 的轨迹。"""
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

    # ---------- 结构化偏差与证据优先裁决 ----------
    def detect_divergence(self, workspace_id: str, agent_id: str) -> DivergenceReport:
        """使用该 Agent 最新一条黑板记录与工作区规范状态比较。"""
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
        """汇总同一 state key 的证据；裸状态声明降级为低置信 Agent inference。"""
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
        """找出被不同 Agent 声明为多个规范化值的 state keys。"""
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

    # ---------- 候选队列、安全门与原子晋升 ----------
    def propose_sop(self, candidate: SOPCandidate) -> tuple[SOPCandidate, SafetyDecision]:
        """按 utility 与安全风险预检候选并持久化；update/delete 必须声明目标版本。"""
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
        """检查候选级状态证据和因果置信度，幂等保持已晋升状态。"""
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
        """记录一次验证；首个合格成功 trial 会在返回前尝试写入 SOP。"""
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
        """重算全部晋升门并执行 create/update/delete；对已晋升候选幂等。"""
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
        # “合格 trial”不仅要 success，还必须有状态验证和外部因果支持。
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
        # 默认阈值均为 1，因此第一个合格 trial 就满足该兼容性命名的历史门控。
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
        # 验证分由因果置信度、trial 覆盖和安全风险三部分组成，并裁剪到 1。
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
        """统一安全门入口；关闭安全门只允许用于隔离的消融数据库。"""
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

    # ---------- 检索与真实执行反馈 ----------
    def retrieve_sops(
        self, query: str, limit: int = 5, query_plan: TaskPlan | None = None,
        *, context_facts: set[str] | None = None,
    ) -> list[RetrievalResult]:
        if not self.config.enable_procedural_memory:
            return []
        sops, _ = self.store.list_active_sops(limit=1_000, offset=0)
        return self.retriever.rank(query, sops, limit, query_plan, context_facts=context_facts)

    def record_sop_outcome(self, sop_id: str, success: bool, preferred_feature: str | None = None, query: str = "") -> SOPVersion:
        """更新 SOP 使用/成功统计，并可对动态检索路由器做一步反馈学习。"""
        updated = self.store.record_sop_outcome(sop_id, success)
        if preferred_feature:
            self.router.learn(query, preferred_feature)
        return updated
