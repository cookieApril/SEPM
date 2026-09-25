#!/usr/bin/env python3
"""Audit the E2 SOP-model-sensitivity matrix without running experiments."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "evaluation_matrix.json"
PLAN_PATH = ROOT / "benchmark-results" / "plans" / "sop-model-sensitivity.jsonl"
TABLE_DIR = ROOT / "benchmark-results" / "unified-v3" / "tables"
REPORT_MD = TABLE_DIR / "e2_readiness_report.md"
REPORT_JSON = TABLE_DIR / "e2_readiness_report.json"

EXPECTED_SOP_MODELS = [
    "gpt-5.6-terra",
    "claude-opus-5",
    "qwen3.8-max",
    "deepseek-v4-flash-0731",
    "kimi-k3",
    "glm-5.2",
    "gemma-4-31b-it",
    "gemma-4-12b-it",
    "qwen3.5-27b",
    "qwen3.5-9b",
    "qwen3.5-2b",
    "qwen3.5-0.8b",
]


def _load_plan() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in PLAN_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _matched_issues(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    by_task: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        by_task[row["task"]][row["sop_model"]].add(str(row["case_id"]))
    for task, by_model in sorted(by_task.items()):
        expected = None
        for model, case_ids in sorted(by_model.items()):
            if expected is None:
                expected = case_ids
                continue
            if case_ids != expected:
                issues.append(
                    {
                        "task": task,
                        "reason": "case_ids differ across SOP models",
                        "case_sets": {name: sorted(values) for name, values in by_model.items()},
                    }
                )
                break
    return issues


def main() -> int:
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    e2 = next(item for item in matrix["benchmarks"] if item["id"] == "sop-model-sensitivity")
    rows = _load_plan()

    distribution = []
    cases_by_task_model: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        cases_by_task_model[(row["task"], row["sop_model"])].add(str(row["case_id"]))
    for (task, sop_model), case_ids in sorted(cases_by_task_model.items()):
        distribution.append(
            {
                "benchmark": task,
                "sop_model": sop_model,
                "case_count": len(case_ids),
                "case_ids": sorted(case_ids),
            }
        )

    invariant_violations = []
    if set(row["mas_framework"] for row in rows) != {"autogen"}:
        invariant_violations.append("MAS is not fixed to autogen")
    if set(row["actor_model"] for row in rows) != {"gpt-5-mini"}:
        invariant_violations.append("actor model is not fixed to gpt-5-mini")
    if set(row["memory_method"] for row in rows) != {"team-memory"}:
        invariant_violations.append("memory_method is not fixed to team-memory")
    if set(row.get("ablation", "full") for row in rows) != {"full"}:
        invariant_violations.append("ablation is not fixed to full")
    if sorted(set(row["sop_model"] for row in rows)) != sorted(EXPECTED_SOP_MODELS):
        invariant_violations.append("SOP model list differs from E2 protocol")

    mab_cases = sorted({str(row["case_id"]) for row in rows if row["task"] == "multiagentbench"})
    blocked_mab = [
        case_id
        for case_id in mab_cases
        if not case_id.startswith("research/")
    ]
    matched_issues = _matched_issues(rows)
    target_cells = sum(len(set(item for item in e2["case_ids"][task])) for task in e2["tasks"]) * len(EXPECTED_SOP_MODELS)

    report = {
        "matrix_path": str(MATRIX_PATH.relative_to(ROOT)),
        "plan_path": str(PLAN_PATH.relative_to(ROOT)),
        "target_cells": target_cells,
        "plan_cells": len(rows),
        "tasks": list(e2["tasks"]),
        "sop_models": list(e2["sop_models"]),
        "fixed_invariants": {
            "mas_framework": sorted(set(row["mas_framework"] for row in rows)),
            "actor_model": sorted(set(row["actor_model"] for row in rows)),
            "memory_method": sorted(set(row["memory_method"] for row in rows)),
            "ablation": sorted(set(row.get("ablation", "full") for row in rows)),
            "seed": sorted(set(row["seed"] for row in rows)),
        },
        "distribution": distribution,
        "case_counts_by_benchmark": dict(Counter(row["task"] for row in rows)),
        "case_counts_by_sop_model": dict(Counter(row["sop_model"] for row in rows)),
        "multiagentbench_research_only": not blocked_mab and bool(mab_cases),
        "blocked_multiagentbench_cases": blocked_mab,
        "matched_case_issues": matched_issues,
        "invariant_violations": invariant_violations,
        "ok": not invariant_violations and not blocked_mab and not matched_issues,
        "notes": [
            "This is a readiness audit only; no E2 experiment cells were executed.",
            "Smoke rows are engineering checks and must not be used as formal E2 evidence.",
        ],
    }

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# E2 SOP-Model Sensitivity Readiness Report",
        "",
        "This report audits the runnable E2 matrix only; it does not contain experiment results.",
        "",
        f"- Target cells: {target_cells}",
        f"- Plan cells: {len(rows)}",
        f"- Fixed MAS: {', '.join(report['fixed_invariants']['mas_framework'])}",
        f"- Fixed actor model: {', '.join(report['fixed_invariants']['actor_model'])}",
        f"- Fixed memory method: {', '.join(report['fixed_invariants']['memory_method'])}",
        f"- MultiAgentBench Research-only: {report['multiagentbench_research_only']}",
        f"- Matched case issues: {len(matched_issues)}",
        f"- Invariant violations: {len(invariant_violations)}",
        "",
        "## Cell Distribution",
        "",
        "| Benchmark | SOP model | Cases |",
        "| --- | --- | ---: |",
    ]
    for item in distribution:
        lines.append(f"| {item['benchmark']} | {item['sop_model']} | {item['case_count']} |")
    if blocked_mab:
        lines.extend(["", "## Blocked MultiAgentBench Cases", ""])
        lines.extend(f"- {case_id}" for case_id in blocked_mab)
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
