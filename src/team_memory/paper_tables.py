"""从统一 ``summary.json`` 生成论文实验表。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


PAPER_BENCHMARK_ORDER = ["alfworld", "webarena", "multiagentbench", "officebench"]
TASK_LABELS = {
    "alfworld": "ALFWorld",
    "webarena": "WebArena",
    "multiagentbench": "MultiAgentBench",
    "officebench": "OfficeBench",
}
MEMORY_LABELS = {
    "no-memory": "No-memory",
    "agent-native-memory": "Agent-native memory",
    "generative-memory": "Generative Memory",
    "metagpt": "MetaGPT",
    "agentverse": "AgentVerse",
    "gmemory": "G-Memory",
    "decentmem": "DecentMem",
    "chatdev": "ChatDev",
    "legomem": "LEGOMem",
    "memp": "Memp",
    "mem0": "mem0",
    "team-memory": "Team Memory (Ours)",
}
MEMORY_ORDER = [
    "no-memory",
    "agent-native-memory",
    "generative-memory",
    "gmemory",
    "mem0",
    "team-memory",
]
MINIMAL_ABLATION_ORDER = [
    "no-extra-components",
    "blackboard-only",
    "sop-only",
    "divergence-only",
    "full",
]


def _select_groups(
    report: dict[str, Any],
    *,
    benchmark: str,
    actor_model: str,
    memory_method: str | None = None,
) -> list[dict[str, Any]]:
    """固定任务 Actor；SOP-Agent 是另一维，绝不能与 Actor 平均。"""
    return [
        group
        for group in report.get("groups", [])
        if group.get("benchmark") == benchmark
        and group.get("actor_model") == actor_model
        and (memory_method is None or group.get("memory_method") == memory_method)
    ]


def _score(group: dict[str, Any], metric: str) -> float | None:
    """兼容新版 metric_stats 和旧版 mean_metrics。"""
    stats = group.get("metric_stats", {}).get(metric)
    if isinstance(stats, dict) and isinstance(stats.get("mean"), (int, float)):
        return float(stats["mean"])
    value = group.get("mean_metrics", {}).get(metric)
    return float(value) if isinstance(value, (int, float)) else None


def _mean_metric(groups: list[dict[str, Any]], metric: str) -> float | None:
    values = [_score(group, metric) for group in groups]
    numeric = [value for value in values if value is not None]
    return fmean(numeric) if numeric else None


def build_main_rows(
    report: dict[str, Any],
    *,
    actor_model: str,
    sop_model: str,
    benchmark: str = "cross-benchmark-generality",
    metric: str = "primary_score",
    tasks: list[str] | None = None,
) -> list[dict[str, Any]]:
    """生成 ``MAS -> Memory -> benchmark columns`` 主性能表。"""
    task_order = tasks or PAPER_BENCHMARK_ORDER
    groups = [
        row
        for row in _select_groups(report, benchmark=benchmark, actor_model=actor_model)
        if row.get("ablation", "full") == "full"
        and (
            row.get("memory_method") != "team-memory"
            or row.get("sop_model") == sop_model
        )
    ]
    scores: dict[tuple[str, str, str], float] = {}
    memory_seen: dict[str, set[str]] = defaultdict(set)
    for group in groups:
        value = _score(group, metric)
        if value is None:
            continue
        mas = group["mas_framework"]
        memory = group["memory_method"]
        scores[(mas, memory, group["task"])] = value
        memory_seen[mas].add(memory)

    rows: list[dict[str, Any]] = []
    for mas in sorted(memory_seen):
        memories = sorted(
            memory_seen[mas],
            key=lambda item: (
                MEMORY_ORDER.index(item) if item in MEMORY_ORDER else len(MEMORY_ORDER),
                item,
            ),
        )
        for memory in memories:
            row: dict[str, Any] = {
                "MAS": mas,
                "Memory method": MEMORY_LABELS.get(memory, memory),
            }
            available: list[float] = []
            for task in task_order:
                value = scores.get((mas, memory, task))
                row[TASK_LABELS.get(task, task)] = value
                if value is not None:
                    available.append(value)
            row["Avg."] = fmean(available) if len(available) == len(task_order) else None
            rows.append(row)
    return rows


def build_ablation_rows(
    report: dict[str, Any],
    *,
    actor_model: str,
    sop_model: str,
    mas_framework: str,
    benchmark: str = "component-ablation",
    metric: str = "primary_score",
    tasks: list[str] | None = None,
) -> list[dict[str, Any]]:
    """生成固定模型、固定 MAS、固定 Team Memory/GEMS 的组件消融表。"""
    task_order = tasks or PAPER_BENCHMARK_ORDER
    groups = [
        row
        for row in _select_groups(
            report,
            benchmark=benchmark,
            actor_model=actor_model,
            memory_method="team-memory",
        )
        if row.get("mas_framework") == mas_framework
        and row.get("sop_model") == sop_model
    ]
    grouped_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_by_variant_task: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    scores: dict[tuple[str, str], float | None] = {}
    for group in groups:
        variant = group.get("ablation", "full")
        grouped_by_variant[variant].append(group)
        grouped_by_variant_task[(variant, group["task"])].append(group)
        scores[(variant, group["task"])] = _score(group, metric)
    found = set(grouped_by_variant)
    variants = [variant for variant in MINIMAL_ABLATION_ORDER if variant in found]
    variants.extend(sorted(found - set(variants)))
    rows: list[dict[str, Any]] = []
    for task in task_order:
        for variant in variants:
            task_groups = grouped_by_variant_task.get((variant, task), [])
            if not task_groups:
                continue
            value = scores.get((variant, task))
            baseline = scores.get(("no-extra-components", task))
            actor_tokens = _mean_metric(task_groups, "actor_prompt_tokens")
            sop_tokens = _mean_metric(task_groups, "sop_prompt_tokens")
            row: dict[str, Any] = {
                "Benchmark": TASK_LABELS.get(task, task),
                "Condition": variant,
            }
            row.update(
                {
                    "Procedural SOP": "On" if variant in {"sop-only", "full"} else "Off",
                    "Blackboard": (
                        "On"
                        if variant in {
                            "blackboard-only",
                            "sop-only",
                            "divergence-only",
                            "full",
                        }
                        else "Off"
                    ),
                    "Divergence alignment": (
                        "On" if variant in {"divergence-only", "full"} else "Off"
                    ),
                    "Official task score": value,
                    "Delta vs no-extra-components": (
                        value - baseline
                        if value is not None
                        and baseline is not None
                        and variant != "no-extra-components"
                        else None
                    ),
                    "SOP reuse success": _mean_metric(task_groups, "sop_reuse_success"),
                    "Divergence recovery": _mean_metric(task_groups, "divergence_recovery"),
                    "Unsafe accepted": _mean_metric(task_groups, "unsafe_accepted"),
                    "Token overhead": (
                        (actor_tokens or 0.0) + (sop_tokens or 0.0)
                        if actor_tokens is not None or sop_tokens is not None
                        else None
                    ),
                    "Latency": _mean_metric(task_groups, "latency_seconds"),
                }
            )
            rows.append(row)
    return rows


def build_sop_model_rows(
    report: dict[str, Any],
    *,
    actor_model: str,
    mas_framework: str = "autogen",
    benchmark: str = "sop-model-sensitivity",
    metric: str = "primary_score",
    default_sop_model: str | None = None,
    tasks: list[str] | None = None,
) -> list[dict[str, Any]]:
    """生成 SOP-Agent 模型敏感性表；Actor、MAS、prompt 与预算保持不变。"""
    task_order = tasks or PAPER_BENCHMARK_ORDER
    groups = [
        row
        for row in _select_groups(
            report,
            benchmark=benchmark,
            actor_model=actor_model,
            memory_method="team-memory",
        )
        if row.get("mas_framework") == mas_framework
        and row.get("ablation", "full") == "full"
    ]
    by_model_task = {(group["sop_model"], group["task"]): group for group in groups}
    scores = {
        key: _score(group, metric)
        for key, group in by_model_task.items()
    }
    default_model = default_sop_model or actor_model
    rows: list[dict[str, Any]] = []
    for sop_model in sorted({model for model, _ in scores}):
        for task in task_order:
            group = by_model_task.get((sop_model, task))
            value = scores.get((sop_model, task))
            default_value = scores.get((default_model, task))
            rows.append(
                {
                    "Benchmark": benchmark,
                    "Task": TASK_LABELS.get(task, task),
                    "MAS": mas_framework,
                    "Actor model": actor_model,
                    "SOP model": sop_model,
                    "Score": value,
                    "Delta vs default SOP model": (
                        value - default_value
                        if value is not None and default_value is not None
                        else None
                    ),
                    "SOP JSON validity": _score(group, "sop_json_valid_rate")
                    if group
                    else None,
                    "Candidate pass rate": _score(group, "sop_candidate_pass_rate")
                    if group
                    else None,
                    "Evidence correctness": _score(group, "evidence_correctness")
                    if group
                    else None,
                    "Safety rejections": _score(group, "sop_safety_rejections")
                    if group
                    else None,
                    "SOP-model cost": (
                        (_score(group, "sop_prompt_tokens") or 0.0)
                        + (_score(group, "sop_completion_tokens") or 0.0)
                        if group
                        and (
                            _score(group, "sop_prompt_tokens") is not None
                            or _score(group, "sop_completion_tokens") is not None
                        )
                        else None
                    ),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build MAS x Memory paper tables")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("benchmark-results/unified-v3/summary.json"),
    )
    parser.add_argument("--actor-model")
    parser.add_argument("--sop-model")
    parser.add_argument("--default-sop-model")
    parser.add_argument("--mas", default="autogen", help="MAS used for the ablation table")
    parser.add_argument("--metric", default="primary_score")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-results/tables"))
    args = parser.parse_args(argv)

    if not args.actor_model or not args.sop_model:
        parser.error("--actor-model and --sop-model are required")
    if not args.summary.is_file():
        parser.error(f"summary not found: {args.summary}")
    report = json.loads(args.summary.read_text(encoding="utf-8"))
    _write_csv(
        args.output_dir / "cross_benchmark_main.csv",
        build_main_rows(
            report,
            actor_model=args.actor_model,
            sop_model=args.sop_model,
            metric=args.metric,
        ),
    )
    _write_csv(
        args.output_dir / "sop_model_sensitivity.csv",
        build_sop_model_rows(
            report,
            actor_model=args.actor_model,
            mas_framework=args.mas,
            metric=args.metric,
            default_sop_model=args.default_sop_model or args.sop_model,
        ),
    )
    _write_csv(
        args.output_dir / "ablation.csv",
        build_ablation_rows(
            report,
            actor_model=args.actor_model,
            sop_model=args.sop_model,
            mas_framework=args.mas,
            metric=args.metric,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
