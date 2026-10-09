"""Deterministic safety gate applied before SOP publication.

Rules protect confirmation, backup, authorization, and irreversible-operation
steps and reject updates that weaken mandatory safeguards. Safety runs before
statistical utility or learned models, so higher success cannot override a violation.
"""

from __future__ import annotations

from collections.abc import Iterable

from .config import MemoryConfig
from .models import (
    ProcedureGraph,
    ProcedureStep,
    ProposalOperation,
    SafetyDecision,
    SOPCandidate,
    SOPVersion,
)


class SafetyValidator:
    """Interpretable deterministic rules applied before LLM or learned validation."""

    CONFIRMATION_WORDS = ("confirm", "approval", "authorize", "确认", "批准", "授权")
    BACKUP_WORDS = ("backup", "snapshot", "备份", "快照")

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self.config = config or MemoryConfig()

    def validate(self, candidate: SOPCandidate, current: SOPVersion | None = None) -> SafetyDecision:
        """Validate a create/update/delete candidate and return risk and violations."""
        violations: list[str] = []
        if (
            current
            and candidate.metadata
            and candidate.operation == ProposalOperation.UPDATE
            and (
                candidate.metadata.context_conditions != current.metadata.context_conditions
                or candidate.metadata.variant_of != current.metadata.variant_of
            )
        ):
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
        # Strict checks target sensitive steps; all steps still participate in safeguard comparison.
        sensitive = [step for step in graph.steps if self._is_sensitive(step) or self._is_irreversible(step)]
        for step in sensitive:
            text = step.instruction.lower()
            if self._is_irreversible(step) and not step.requires_confirmation:
                violations.append(f"irreversible step {step.step_id} lacks confirmation")
            if ("migration" in text or "迁移" in text) and not any(
                self._contains_any(item.instruction.lower(), self.BACKUP_WORDS)
                for item in graph.steps
            ):
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
        """Treat explicit safety-critical steps or configured keyword matches as sensitive."""
        text = f"{step.action_type} {step.instruction}".lower()
        return step.safety_critical or self._contains_any(text, self.config.safety_keywords)

    @staticmethod
    def _contains_any(text: str, words: Iterable[str]) -> bool:
        return any(word in text for word in words)

    @staticmethod
    def _is_irreversible(step: ProcedureStep) -> bool:
        """Identify irreversible actions such as payments, deletions, and transfers."""
        text = f"{step.action_type} {step.instruction}".lower()
        markers = ("pay", "payment", "delete", "drop", "transfer", "付款", "支付", "删除", "转账")
        return any(marker in text for marker in markers)

    def _preserves_mandatory_steps(self, old: ProcedureGraph | None, new: ProcedureGraph) -> bool:
        """Conservatively check that an update preserves safeguards and confirmations."""
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
