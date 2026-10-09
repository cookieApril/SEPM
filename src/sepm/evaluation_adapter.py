"""Shared input/output contract for external benchmark adapters."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvaluationContext:
    """Nonsecret experiment identity passed to an isolated Conda process."""

    benchmark: str
    task: str
    memory_method: str
    actor_model: str
    sop_model: str
    mas_framework: str
    seed: int
    ablation: str
    case_id: str | None
    output_path: Path

    @classmethod
    def from_environment(cls) -> EvaluationContext:
        required = {
            "benchmark": "SEPM_EVAL_BENCHMARK",
            "task": "SEPM_EVAL_TASK",
            "memory_method": "SEPM_EVAL_METHOD",
            "actor_model": "SEPM_EVAL_ACTOR_MODEL",
            "sop_model": "SEPM_EVAL_SOP_MODEL",
            "mas_framework": "SEPM_EVAL_MAS",
            "seed": "SEPM_EVAL_SEED",
            "output_path": "SEPM_EVAL_OUTPUT",
        }
        missing = [variable for variable in required.values() if not os.environ.get(variable)]
        if missing:
            raise RuntimeError(f"missing evaluation environment variables: {missing}")
        raw_case = os.environ.get("SEPM_EVAL_CASE_ID")
        return cls(
            benchmark=os.environ[required["benchmark"]],
            task=os.environ[required["task"]],
            memory_method=os.environ[required["memory_method"]],
            actor_model=os.environ[required["actor_model"]],
            sop_model=os.environ[required["sop_model"]],
            mas_framework=os.environ[required["mas_framework"]],
            seed=int(os.environ[required["seed"]]),
            ablation=os.environ.get("SEPM_EVAL_ABLATION", "full"),
            case_id=raw_case or None,
            output_path=Path(os.environ[required["output_path"]]),
        )

    def write_result(
        self,
        metrics: dict[str, float],
        *,
        cases: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Atomically write standard JSON for aggregation; API keys are forbidden."""
        forbidden = {"api_key", "token", "secret", "authorization"}
        metadata = metadata or {}
        if forbidden.intersection(key.lower() for key in metadata):
            raise ValueError("secrets are forbidden in evaluation metadata")
        payload = {
            "context": {**asdict(self), "output_path": str(self.output_path)},
            "metrics": metrics,
            "cases": cases or [],
            "metadata": metadata,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.output_path)
