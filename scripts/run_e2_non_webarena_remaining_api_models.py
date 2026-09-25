#!/usr/bin/env python3
"""Run non-WebArena E2 cells for remaining API/relay SOP models."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import run_e2_single_sop_model as single


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "benchmark-results" / "unified-v3" / "tables"
LOG_DIR = ROOT / "benchmark-results" / "unified-v3" / "logs"
DB_DIR = ROOT / "benchmark-results" / "unified-v3" / "e2-non-webarena-api-dbs"
STATE_DIR = ROOT / "benchmark-results" / "unified-v3" / "e2-non-webarena-api-states"
SUMMARY_MD = TABLE_DIR / "e2_non_webarena_api_models_summary.md"
QUALITY_JSON = TABLE_DIR / "e2_non_webarena_api_models_quality.json"
RUN_LOG = LOG_DIR / "e2_non_webarena_api_models_runner.log"

TARGET_MODELS = ["gpt-5.6-terra", "qwen3.8-max", "glm-5.2", "claude-opus-5", "kimi-k3"]
TASKS = ("alfworld", "multiagentbench", "officebench")
EXPECTED_CASES_BY_TASK = {"alfworld": 12, "multiagentbench": 12, "officebench": 24}


def _target_models() -> list[str]:
    raw = os.environ.get("E2_NON_WEBARENA_API_MODELS")
    if not raw:
        return TARGET_MODELS
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in requested if item not in TARGET_MODELS]
    if unknown:
        raise RuntimeError(f"unknown E2 non-WebArena API model(s): {unknown}")
    return requested


def _trusted_preflight_models() -> set[str]:
    raw = os.environ.get("E2_ASSUME_PREFLIGHT_OK_MODELS", "")
    return {item.strip() for item in raw.split(",") if item.strip()}


def _load_plan(models: list[str]) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in single.PLAN_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [
        row
        for row in rows
        if row["benchmark"] == "sop-model-sensitivity"
        and row["task"] in TASKS
        and row["task"] != "webarena"
        and row["sop_model"] in models
        and row["mas_framework"] == "autogen"
        and row["actor_model"] == "gpt-5-mini"
        and row["memory_method"] == "team-memory"
        and row.get("ablation", "full") == "full"
        and row["seed"] == 0
    ]
    cases_by_model_task: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in selected:
        cases_by_model_task[(row["sop_model"], row["task"])].add(str(row["case_id"]))

    issues = []
    for task, expected in EXPECTED_CASES_BY_TASK.items():
        expected_set: set[str] | None = None
        for model in models:
            values = cases_by_model_task[(model, task)]
            if len(values) != expected:
                issues.append({"sop_model": model, "task": task, "expected": expected, "got": len(values)})
            if expected_set is None:
                expected_set = values
            elif values != expected_set:
                issues.append({"task": task, "sop_model": model, "reason": "unmatched case set"})
    if issues:
        raise RuntimeError(f"matched non-WebArena E2 plan issue: {json.dumps(issues, ensure_ascii=False)}")

    task_order = {task: index for index, task in enumerate(TASKS)}
    return sorted(selected, key=lambda row: (models.index(row["sop_model"]), task_order[row["task"]], str(row["case_id"])))


def _write_outputs(
    rows: list[dict[str, Any]],
    plan: list[dict[str, Any]],
    preflight: list[dict[str, Any]],
    unavailable: list[dict[str, Any]],
    *,
    stopped: dict[str, Any] | None = None,
    skipped_existing: int = 0,
) -> None:
    completed = {(row["benchmark"], row["sop_model"], row["case_id"]) for row in rows}
    missing = [
        {"benchmark": row["task"], "sop_model": row["sop_model"], "case_id": str(row["case_id"])}
        for row in plan
        if (row["task"], row["sop_model"], str(row["case_id"])) not in completed
    ]
    score_sources = Counter(row["score_source"] for row in rows)
    by_model_task: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_model: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_model_task[(row["sop_model"], row["benchmark"])].append(row["official_score"])
        by_model[row["sop_model"]].append(row["official_score"])

    models = [model for model in TARGET_MODELS if any(row["sop_model"] == model for row in rows)]
    model_task_means = [
        {
            "sop_model": model,
            "benchmark": task,
            "cases": len(by_model_task[(model, task)]),
            "mean_score": sum(by_model_task[(model, task)]) / len(by_model_task[(model, task)]),
        }
        for model in models
        for task in TASKS
        if by_model_task[(model, task)]
    ]
    model_means = [
        {
            "sop_model": model,
            "cases": len(by_model[model]),
            "overall_mean_score": sum(by_model[model]) / len(by_model[model]),
        }
        for model in models
        if by_model[model]
    ]

    matched_case_issues = []
    plan_models = sorted({row["sop_model"] for row in plan}, key=TARGET_MODELS.index)
    for task, expected in EXPECTED_CASES_BY_TASK.items():
        expected_set: set[str] | None = None
        for model in plan_models:
            values = {str(row["case_id"]) for row in plan if row["sop_model"] == model and row["task"] == task}
            if len(values) != expected:
                matched_case_issues.append({"benchmark": task, "sop_model": model, "expected": expected, "got": len(values)})
            if expected_set is None:
                expected_set = values
            elif values != expected_set:
                matched_case_issues.append({"benchmark": task, "sop_model": model, "reason": "unmatched case set"})

    lines = [
        "# E2 Non-WebArena API Models Summary",
        "",
        "This table covers only non-WebArena formal E2 cells for remaining API/relay SOP models.",
        "",
    ]
    if unavailable:
        lines.extend(["## Unavailable Models", ""])
        for item in unavailable:
            lines.append(f"- `{item['model_id']}`: {item.get('error', 'preflight failed')}")
        lines.append("")
    if stopped:
        lines.extend(["## Stop Status", "", f"- `{json.dumps(stopped, ensure_ascii=False)}`", ""])
    lines.extend(
        [
            "## Per-Cell Results",
            "",
            "| SOP model | Benchmark | Case ID | Official score | Score source | SOP parse success | SOP reuse count | SOP reuse success | Token usage | Latency | Result path |",
            "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in sorted(rows, key=lambda item: (item["sop_model"], item["benchmark"], item["case_id"])):
        lines.append(
            "| {sop_model} | {benchmark} | {case_id} | {score:.3f} | {source} | {parse} | {reuse_count} | {reuse_success} | {tokens} | {latency} | {path} |".format(
                sop_model=row["sop_model"],
                benchmark=row["benchmark"],
                case_id=row["case_id"],
                score=row["official_score"],
                source=row["score_source"],
                parse=single._format_num(row["sop_parse_success"]),
                reuse_count=single._format_num(row["sop_reuse_count"]),
                reuse_success=single._format_num(row["sop_reuse_success"]),
                tokens=single._format_num(row["token_usage"]),
                latency=single._format_num(row["latency"]),
                path=row["result_path"],
            )
        )
    lines.extend(["", "## Means", "", "| SOP model | Benchmark | Cases | Mean official score |", "| --- | --- | ---: | ---: |"])
    for item in model_task_means:
        lines.append(f"| {item['sop_model']} | {item['benchmark']} | {item['cases']} | {item['mean_score']:.3f} |")
    lines.extend(["", "| SOP model | Cases | Overall mean score |", "| --- | ---: | ---: |"])
    for item in model_means:
        lines.append(f"| {item['sop_model']} | {item['cases']} | {item['overall_mean_score']:.3f} |")

    quality = {
        "expected_cells": len(plan),
        "completed_cells": len(rows),
        "skipped_existing_valid_results": skipped_existing,
        "unavailable_models": unavailable,
        "missing_cells": missing,
        "schema_violations": [] if stopped is None or stopped.get("reason") != "result_validation_failed" else [stopped],
        "matched_case_issues": matched_case_issues,
        "runtime_failures": [] if stopped is None or stopped.get("reason") == "result_validation_failed" else [stopped],
        "score_source_distribution": dict(score_sources),
        "model_benchmark_mean_scores": model_task_means,
        "model_overall_mean_scores": model_means,
        "per_cell_result_paths": rows,
        "preflight": preflight,
        "stopped": stopped,
        "notes": [
            "WebArena is explicitly excluded because AWS EC2 is stopped.",
            "Relay/API SOP models are not labeled as local models.",
            "Official scored 0.0 rows are accepted results.",
            "No failed run, sidecar, or partial row is counted.",
            "Official evaluator scoring code is not modified by this runner.",
            "Execution is serial to avoid OfficeBench Docker and MARBLE run_dir conflicts.",
        ],
    }
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    QUALITY_JSON.write_text(json.dumps(quality, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if {row["sop_model"] for row in plan} == {"glm-5.2"}:
        glm_summary = TABLE_DIR / "e2_glm-5.2_non_webarena_summary.md"
        glm_quality = TABLE_DIR / "e2_glm-5.2_non_webarena_quality.json"
        glm_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        glm_quality.write_text(json.dumps(quality, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if {row["sop_model"] for row in plan}.issubset({"claude-opus-5", "kimi-k3"}):
        ck_summary = TABLE_DIR / "e2_claude_kimi_non_webarena_summary.md"
        ck_quality = TABLE_DIR / "e2_claude_kimi_non_webarena_quality.json"
        ck_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        ck_quality.write_text(json.dumps(quality, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    os.chdir(ROOT)
    single._load_dotenv()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    single.DB_DIR = DB_DIR
    single.STATE_DIR = STATE_DIR
    single.RUN_LOG = RUN_LOG
    RUN_LOG.write_text("", encoding="utf-8")

    preflight: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    available: list[str] = []
    target_models = _target_models()
    trusted_preflight = _trusted_preflight_models()
    for model in target_models:
        print(f"PREFLIGHT {model}", flush=True)
        if model in trusted_preflight:
            checked = {
                "model_id": model,
                "api_model": model,
                "ok": True,
                "json_parse_ok": True,
                "cost_accounting_ok": True,
                "trusted_external_preflight": True,
            }
        else:
            checked = single._preflight_model(model)
        preflight.append(checked)
        if checked.get("ok"):
            available.append(model)
            print(f"OK {model}", flush=True)
        else:
            unavailable.append(checked)
            print(f"UNAVAILABLE {model}: {checked.get('error')}", flush=True)

    plan = _load_plan(available) if available else []
    if any(row["task"] == "webarena" for row in plan):
        raise RuntimeError("internal filter error: WebArena row selected while AWS is stopped")

    rows: list[dict[str, Any]] = []
    skipped_existing = 0
    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = "src"

    for index, row in enumerate(plan, start=1):
        label = f"{index}/{len(plan)} {row['task']} autogen team-memory {row['case_id']} sop={row['sop_model']}"
        try:
            existing = single._latest_existing_result(row)
            if existing is not None:
                checked = single._validate_result(existing, row)
                skipped_existing += 1
                print(f"SKIP {label} score={checked['official_score']}", flush=True)
            else:
                print(f"RUN {label}", flush=True)
                result = single._run_cell(row, env_base)
                checked = single._validate_result(result, row)
            rows.append(checked)
            _write_outputs(rows, plan, preflight, unavailable, skipped_existing=skipped_existing)
            print(f"OK {label} score={checked['official_score']}", flush=True)
        except Exception as exc:
            reason = "result_validation_failed" if "context mismatch" in str(exc) or "missing metrics" in str(exc) else "cell_failed"
            stopped = {"cell": label, "reason": reason, "error": str(exc)}
            _write_outputs(rows, plan, preflight, unavailable, stopped=stopped, skipped_existing=skipped_existing)
            print(json.dumps(stopped, indent=2, ensure_ascii=False), flush=True)
            return 2

    _write_outputs(rows, plan, preflight, unavailable, skipped_existing=skipped_existing)
    single._write_available_merged_outputs()
    print(
        json.dumps(
            {
                "completed": len(rows),
                "expected": len(plan),
                "available_models": available,
                "unavailable_models": [item["model_id"] for item in unavailable],
                "skipped_existing": skipped_existing,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
