"""只展开 Team Memory 最小组件条件的命令行入口。

主结果矩阵默认只运行 ``full``。本 runner 会把指定 benchmark 的 ablation 维度替换为
无新增部件、仅 SOP、仅任务理解偏差机制或完整方法，同时固定 memory method 为
``team-memory``，防止把 G-Memory/No-memory 错误地乘上 Team Memory 内部条件。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..evaluation_runner import build_report, expand_matrix, load_env_file, load_matrix, run_matrix
from .runtime import ABLATION_OVERRIDES


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Team Memory ablations")
    parser.add_argument("action", choices=("matrix", "run", "report"))
    parser.add_argument("--config", type=Path, default=Path("evaluation_matrix.json"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--benchmark", default="component-ablation")
    parser.add_argument("--task")
    parser.add_argument("--actor-model")
    parser.add_argument("--sop-model")
    parser.add_argument("--mas")
    parser.add_argument("--variant", action="append", choices=sorted(ABLATION_OVERRIDES))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--case-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    load_env_file(args.env_file)
    config = load_matrix(args.config)
    if args.action == "report":
        print(json.dumps(build_report(config), ensure_ascii=False, indent=2))
        return 0

    selected = args.variant or list(ABLATION_OVERRIDES)
    benchmark = next(
        (row for row in config["benchmarks"] if row["id"] == args.benchmark),
        None,
    )
    if benchmark is None:
        raise SystemExit(f"unknown benchmark: {args.benchmark}")
    benchmark["ablations"] = selected
    cells = expand_matrix(
        config,
        benchmark_filter=args.benchmark,
        task_filter=args.task,
        method_filter="team-memory",
        actor_model_filter=args.actor_model,
        sop_model_filter=args.sop_model,
        mas_filter=args.mas,
        seed_filter=args.seed,
        case_id=args.case_id,
    )
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be >= 1")
        cells = cells[: args.limit]
    if args.action == "matrix":
        for cell in cells:
            print(json.dumps({"cell_key": cell.key, **cell.__dict__}, ensure_ascii=False))
        return 0
    summary = run_matrix(
        config,
        cells,
        resume=not args.no_resume,
        rerun_failed=args.rerun_failed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary.get("failed", 0) or summary.get("not_configured", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
