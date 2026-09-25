"""外部 benchmark 适配器共享的输入/输出契约。"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvaluationContext:
    """runner 通过环境变量传给隔离 Conda 进程的非秘密实验身份。"""

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
    def from_environment(cls) -> "EvaluationContext":
        required = {
            "benchmark": "TEAM_MEMORY_EVAL_BENCHMARK",
            "task": "TEAM_MEMORY_EVAL_TASK",
            "memory_method": "TEAM_MEMORY_EVAL_METHOD",
            "actor_model": "TEAM_MEMORY_EVAL_ACTOR_MODEL",
            "sop_model": "TEAM_MEMORY_EVAL_SOP_MODEL",
            "mas_framework": "TEAM_MEMORY_EVAL_MAS",
            "seed": "TEAM_MEMORY_EVAL_SEED",
            "output_path": "TEAM_MEMORY_EVAL_OUTPUT",
        }
        missing = [variable for variable in required.values() if not os.environ.get(variable)]
        if missing:
            raise RuntimeError(f"missing evaluation environment variables: {missing}")
        raw_case = os.environ.get("TEAM_MEMORY_EVAL_CASE_ID")
        return cls(
            benchmark=os.environ[required["benchmark"]],
            task=os.environ[required["task"]],
            memory_method=os.environ[required["memory_method"]],
            actor_model=os.environ[required["actor_model"]],
            sop_model=os.environ[required["sop_model"]],
            mas_framework=os.environ[required["mas_framework"]],
            seed=int(os.environ[required["seed"]]),
            ablation=os.environ.get("TEAM_MEMORY_EVAL_ABLATION", "full"),
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
        """原子写出 runner 可汇总的标准 JSON；不得传入 API key。"""
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
