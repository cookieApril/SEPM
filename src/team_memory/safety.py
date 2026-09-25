"""SOP 发布前的确定性安全前置门。

规则保护确认、备份、授权和不可逆操作步骤，并检查更新是否弱化旧版本中的强制安全
步骤。安全门先于统计收益与学习模型运行，因此更高成功率不能覆盖安全违规。
"""

from __future__ import annotations

from collections.abc import Iterable

from .config import MemoryConfig
from .models import ProcedureGraph, ProcedureStep, ProposalOperation, SOPCandidate, SOPVersion, SafetyDecision


class SafetyValidator:
    """在 LLM/学习型验证之前运行的可解释、确定性规则集合。"""

    CONFIRMATION_WORDS = ("confirm", "approval", "authorize", "确认", "批准", "授权")
    BACKUP_WORDS = ("backup", "snapshot", "备份", "快照")

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self.config = config or MemoryConfig()

    def validate(self, candidate: SOPCandidate, current: SOPVersion | None = None) -> SafetyDecision:
        """验证 create/update/delete 候选，返回是否允许、风险分和具体违规。"""
        violations: list[str] = []
        if current and candidate.metadata and candidate.operation == ProposalOperation.UPDATE:
            if (candidate.metadata.context_conditions != current.metadata.context_conditions
                    or candidate.metadata.variant_of != current.metadata.variant_of):
                violations.append("context-specific changes must create a variant, not overwrite the parent")
        graph = candidate.procedure
        if candidate.operation == ProposalOperation.DELETE:
            if current and current.metadata.safety_class in {"sensitive", "critical"}:
                violations.append("safety-critical SOPs cannot be deleted; supersede with a newer safe version")
            risk = 1.0 if violations else 0.25
            if risk >= self.config.safety_risk_threshold and not violations:
                violations.append("risk score exceeds safety threshold")
            return SafetyDecision(
                allowed=not violations,
                risk_score=risk,
                risk_threshold=self.config.safety_risk_threshold,
                violations=violations,
                mandatory_steps_preserved=not violations,
            )
        if graph is None:
            return SafetyDecision(
                allowed=False,
                risk_score=1.0,
                risk_threshold=self.config.safety_risk_threshold,
                violations=["missing procedure"],
            )
        # 仅敏感步骤进入昂贵/严格规则；普通步骤仍参与“是否保留旧安全步骤”的比较。
        sensitive = [step for step in graph.steps if self._is_sensitive(step) or self._is_irreversible(step)]
        for step in sensitive:
            text = step.instruction.lower()
            if self._is_irreversible(step) and not step.requires_confirmation:
                violations.append(f"irreversible step {step.step_id} lacks confirmation")
            if "migration" in text or "迁移" in text:
                if not any(self._contains_any(item.instruction.lower(), self.BACKUP_WORDS) for item in graph.steps):
                    violations.append("migration procedure lacks a backup/snapshot step")
        preserved = self._preserves_mandatory_steps(current.procedure if current else None, graph)
        if not preserved:
            violations.append("update removed or weakened a mandatory safety step")
        risk = min(1.0, 0.15 * len(sensitive) + 0.55 * len(violations))
        allowed = not violations and risk < self.config.safety_risk_threshold
        if not violations and not allowed:
            violations.append("risk score exceeds safety threshold")
        return SafetyDecision(
            allowed=allowed,
            risk_score=risk,
            risk_threshold=self.config.safety_risk_threshold,
            violations=violations,
            mandatory_steps_preserved=preserved,
        )

    def _is_sensitive(self, step: ProcedureStep) -> bool:
        """显式 safety_critical 或命中配置关键词的步骤均视为敏感。"""
        text = f"{step.action_type} {step.instruction}".lower()
        return step.safety_critical or self._contains_any(text, self.config.safety_keywords)

    @staticmethod
    def _contains_any(text: str, words: Iterable[str]) -> bool:
        return any(word in text for word in words)

    @staticmethod
    def _is_irreversible(step: ProcedureStep) -> bool:
        """识别付款、删除、转账等需要确认的不可逆动作。"""
        text = f"{step.action_type} {step.instruction}".lower()
        markers = ("pay", "payment", "delete", "drop", "transfer", "付款", "支付", "删除", "转账")
        return any(marker in text for marker in markers)

    def _preserves_mandatory_steps(self, old: ProcedureGraph | None, new: ProcedureGraph) -> bool:
        """保守检查更新是否仍保留旧版本的安全关键语义和确认动作。"""
        if old is None:
            return True
        mandatory = [step for step in old.steps if step.safety_critical or step.requires_confirmation]
        new_steps = {step.step_id: step for step in new.steps}
        for step in mandatory:
            replacement = new_steps.get(step.step_id)
            if replacement is None:
                replacement = next((item for item in new.steps if item.instruction.casefold() == step.instruction.casefold()), None)
            if replacement is None:
                return False
            if step.safety_critical and not replacement.safety_critical:
                return False
            if step.requires_confirmation and not replacement.requires_confirmation:
                return False
            if not set(step.preconditions).issubset(replacement.preconditions):
                return False
            if step.instruction.casefold() != replacement.instruction.casefold():
                return False
        return True
