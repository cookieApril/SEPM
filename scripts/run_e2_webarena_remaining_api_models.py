#!/usr/bin/env python3
"""Run the WebArena-only E2 block for remaining API/relay SOP models."""

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
DB_DIR = ROOT / "benchmark-results" / "unified-v3" / "e2-webarena-api-dbs"
STATE_DIR = ROOT / "benchmark-results" / "unified-v3" / "e2-webarena-api-states"
SUMMARY_MD = TABLE_DIR / "e2_webarena_api_models_aws_summary.md"
QUALITY_JSON = TABLE_DIR / "e2_webarena_api_models_aws_quality.json"
RUN_LOG = LOG_DIR / "e2_webarena_api_models_aws_runner.log"

TARGET_MODELS = [
    "gpt-5.6-terra",
    "claude-opus-5",
    "qwen3.8-max",
    "kimi-k3",
    "glm-5.2",
]
EXPECTED_CASES = 24


def _target_models() -> list[str]:
    raw = os.environ.get("E2_WEBARENA_API_MODELS")
    if not raw:
        return TARGET_MODELS
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in requested if item not in TARGET_MODELS]
    if unknown:
        raise RuntimeError(f"unknown E2 WebArena API model(s): {unknown}")
    return requested


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
        and row["task"] == "webarena"
        and row["sop_model"] in models
        and row["mas_framework"] == "autogen"
        and row["actor_model"] == "gpt-5-mini"
        and row["memory_method"] == "team-memory"
        and row.get("ablation", "full") == "full"
        and row["seed"] == 0
    ]
    cases_by_model: dict[str, set[str]] = defaultdict(set)
    for row in selected:
        cases_by_model[row["sop_model"]].add(str(row["case_id"]))
    issues = []
    expected_set: set[str] | None = None
    for model in models:
        values = cases_by_model[model]
        if len(values) != EXPECTED_CASES:
            issues.append({"sop_model": model, "expected": EXPECTED_CASES, "got": len(values)})
        if expected_set is None:
            expected_set = values
        elif values != expected_set:
            issues.append({"sop_model": model, "reason": "unmatched WebArena case set"})
    if issues:
        raise RuntimeError(f"matched WebArena plan issue: {json.dumps(issues, ensure_ascii=False)}")
    return sorted(selected, key=lambda row: (models.index(row["sop_model"]), str(row["case_id"])))


def _write_outputs(
    rows: list[dict[str, Any]],
    plan: list[dict[str, Any]],
    preflight: list[dict[str, Any]],
    unavailable: list[dict[str, Any]],
    *,
    stopped: dict[str, Any] | None = None,
    skipped_existing: int = 0,
) -> None:
    completed = {(row["sop_model"], row["case_id"]) for row in rows}
    missing = [
        {"sop_model": row["sop_model"], "case_id": str(row["case_id"])}
        for row in plan
        if (row["sop_model"], str(row["case_id"])) not in completed
    ]
    score_sources = Counter(row["score_source"] for row in rows)
    by_model: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_model[row["sop_model"]].append(row["official_score"])
    means = [
        {
            "sop_model": model,
            "benchmark": "webarena",
            "cases": len(scores),
            "mean_score": sum(scores) / len(scores),
        }
        for model, scores in sorted(by_model.items())
    ]
    matched_case_issues = []
    expected_sets: dict[str, set[str]] = defaultdict(set)
    for row in plan:
        expected_sets[row["sop_model"]].add(str(row["case_id"]))
    if expected_sets:
        first = next(iter(expected_sets.values()))
        for model, values in expected_sets.items():
            if len(values) != EXPECTED_CASES or values != first:
                matched_case_issues.append({"sop_model": model, "case_count": len(values)})

    lines = [
        "# E2 WebArena API Models AWS Summary",
        "",
        "This table covers only the WebArena/AWS cells for remaining API/relay SOP models.",
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
            "| SOP model | Benchmark | Case ID | Official score | Score source | Result path |",
            "| --- | --- | --- | ---: | --- | --- |",
        ]
    )
    for row in sorted(rows, key=lambda item: (item["sop_model"], item["case_id"])):
        lines.append(
            f"| {row['sop_model']} | webarena | {row['case_id']} | {row['official_score']:.3f} | {row['score_source']} | {row['result_path']} |"
        )
    lines.extend(["", "## Means", "", "| SOP model | Cases | WebArena mean official score |", "| --- | ---: | ---: |"])
    for item in means:
        lines.append(f"| {item['sop_model']} | {item['cases']} | {item['mean_score']:.3f} |")

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
        "model_benchmark_mean_scores": means,
        "model_overall_mean_scores": [
            {"sop_model": item["sop_model"], "cases": item["cases"], "overall_mean_score": item["mean_score"]}
            for item in means
        ],
        "per_cell_result_paths": rows,
        "preflight": preflight,
        "stopped": stopped,
        "notes": [
            "This is a WebArena-only AWS block, not a full E2 model subset.",
            "Official scored 0.0 rows are accepted results.",
            "No failed run, sidecar, or partial row is counted.",
            "Official evaluator scoring code is not modified by this runner.",
        ],
    }
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    QUALITY_JSON.write_text(json.dumps(quality, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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

    target_models = _target_models()
    preflight: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    available: list[str] = []
    for model in target_models:
        print(f"PREFLIGHT {model}", flush=True)
        checked = single._preflight_model(model)
        preflight.append(checked)
        if checked.get("ok"):
            available.append(model)
            print(f"OK {model}", flush=True)
        else:
            unavailable.append(checked)
            print(f"UNAVAILABLE {model}: {checked.get('error')}", flush=True)

    plan = _load_plan(available) if available else []
    rows: list[dict[str, Any]] = []
    skipped_existing = 0
    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = "src"
    env_base.setdefault("WEBARENA_PLAYWRIGHT_TIMEOUT_MS", "90000")

    for index, row in enumerate(plan, start=1):
        label = f"{index}/{len(plan)} webarena autogen team-memory {row['case_id']} sop={row['sop_model']}"
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
            stopped = {"cell": label, "reason": "cell_failed", "error": str(exc)}
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
