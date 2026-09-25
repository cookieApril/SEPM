#!/usr/bin/env python3
"""Run the bounded 8-cell E2 SOP-model smoke suite."""

from __future__ import annotations

import hashlib
import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "benchmark-results" / "plans" / "sop-model-sensitivity.jsonl"
RESULT_DIR = ROOT / "benchmark-results" / "unified-v3" / "results"
TABLE_DIR = ROOT / "benchmark-results" / "unified-v3" / "tables"
LOG_DIR = ROOT / "benchmark-results" / "unified-v3" / "logs"
DB_DIR = ROOT / "benchmark-results" / "unified-v3" / "e2-smoke-dbs"
STATE_DIR = ROOT / "benchmark-results" / "unified-v3" / "e2-smoke-states"
SUMMARY_MD = TABLE_DIR / "e2_smoke_summary.md"
QUALITY_JSON = TABLE_DIR / "e2_smoke_quality.json"
RUN_LOG = LOG_DIR / "e2_smoke_runner.log"

SMOKE_CASES = {
    "alfworld": "data/alfworld/json_2.1.1/valid_unseen/pick_and_place_simple-SoapBottle-None-Toilet-424/trial_T20190907_004321_405868/game.tw-pddl",
    "webarena": "495",
    "officebench": "2-39/1",
    "multiagentbench": "research/47",
}
SMOKE_SOP_MODELS = ["qwen3.5-2b", "qwen3.5-0.8b"]
EXPECTED_CELLS = len(SMOKE_CASES) * len(SMOKE_SOP_MODELS)
JSON_RESPONSE_FORMAT_MODELS = {
    "qwen3.5-9b",
    "qwen3.5-27b",
    "gemma-4-12b-it",
    "gemma-4-31b-it",
    "deepseek-v4-flash-0731",
}


def _configure_outputs(args: argparse.Namespace) -> None:
    global SMOKE_SOP_MODELS, EXPECTED_CELLS, SUMMARY_MD, QUALITY_JSON, DB_DIR, STATE_DIR, RUN_LOG
    if args.sop_models:
        SMOKE_SOP_MODELS = [item.strip() for item in args.sop_models.split(",") if item.strip()]
    EXPECTED_CELLS = len(SMOKE_CASES) * len(SMOKE_SOP_MODELS)
    if args.summary:
        SUMMARY_MD = ROOT / args.summary
    if args.quality:
        QUALITY_JSON = ROOT / args.quality
    if args.db_dir:
        DB_DIR = ROOT / args.db_dir
    if args.state_dir:
        STATE_DIR = ROOT / args.state_dir
    if args.run_log:
        RUN_LOG = ROOT / args.run_log


def _load_dotenv() -> None:
    source = ROOT / ".env"
    if not source.exists():
        return
    for raw in source.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        os.environ.setdefault(key, value)


def _safe(value: str) -> str:
    cleaned = value.replace("/", "-").replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in cleaned)[:180]


def _stable_tag(row: dict[str, Any]) -> str:
    raw = "|".join(
        str(row[key])
        for key in (
            "benchmark",
            "task",
            "memory_method",
            "actor_model",
            "sop_model",
            "mas_framework",
            "seed",
            "case_id",
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _model_config(model_id: str) -> dict[str, Any]:
    matrix = json.loads((ROOT / "evaluation_matrix.json").read_text(encoding="utf-8"))
    return next(model for model in matrix["models"] if model["id"] == model_id)


def _model_matches(expected_id: str, observed: Any) -> bool:
    if observed is None:
        return False
    observed_text = str(observed)
    if observed_text == expected_id:
        return True
    return observed_text == str(_model_config(expected_id)["api_model"])


def _model_connection(model_id: str) -> dict[str, str]:
    model = _model_config(model_id)
    base_url = os.environ.get(str(model["base_url_env"]), model.get("base_url_default", ""))
    if not base_url:
        raise RuntimeError(f"{model_id}: missing base URL")
    return {
        "model_id": model_id,
        "api_model": str(model["api_model"]),
        "base_url": base_url,
        "api_key": os.environ.get(str(model.get("key_env", "LOCAL_LLM_API_KEY")), "local-no-key") or "local-no-key",
    }


def _preflight_model(model_id: str) -> dict[str, Any]:
    connection = _model_connection(model_id)
    client = OpenAI(
        api_key=connection["api_key"],
        base_url=connection["base_url"],
        timeout=120,
        max_retries=0,
    )
    prompt = (
        "Return only valid JSON for one conservative SOP candidate with keys "
        "title, applicability, exclusions, atomic_steps, safety_constraints, and cost_estimate."
    )
    started = time.perf_counter()
    kwargs: dict[str, Any] = {
        "model": connection["api_model"],
        "messages": [
            {"role": "system", "content": "You emit strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 512,
    }
    if model_id in JSON_RESPONSE_FORMAT_MODELS:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    text = response.choices[0].message.content or ""
    parsed = json.loads(text)
    required = {"title", "applicability", "exclusions", "atomic_steps", "safety_constraints", "cost_estimate"}
    missing = sorted(required - set(parsed))
    if missing:
        raise RuntimeError(f"{model_id}: JSON smoke missing keys {missing}")
    usage = getattr(response, "usage", None)
    return {
        "model_id": model_id,
        "api_model": connection["api_model"],
        "base_url": connection["base_url"],
        "ok": True,
        "json_parse_ok": True,
        "latency_seconds": time.perf_counter() - started,
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        },
        "cost_accounting_ok": usage is not None,
    }


def _load_smoke_plan() -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in PLAN_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = []
    for task, case_id in SMOKE_CASES.items():
        for model in SMOKE_SOP_MODELS:
            matches = [
                row
                for row in rows
                if row["task"] == task
                and row["case_id"] == case_id
                and row["sop_model"] == model
                and row["mas_framework"] == "autogen"
                and row["actor_model"] == "gpt-5-mini"
                and row["memory_method"] == "team-memory"
                and row.get("ablation", "full") == "full"
                and row["seed"] == 0
            ]
            if len(matches) > 1:
                raise RuntimeError(f"expected at most one smoke cell for {task} {case_id} {model}, got {len(matches)}")
            if matches:
                selected.append(matches[0])
            else:
                selected.append(
                    {
                        "benchmark": "sop-model-sensitivity",
                        "task": task,
                        "memory_method": "team-memory",
                        "actor_model": "gpt-5-mini",
                        "sop_model": model,
                        "mas_framework": "autogen",
                        "seed": 0,
                        "ablation": "full",
                        "case_id": case_id,
                    }
                )
    if len(selected) != EXPECTED_CELLS:
        raise RuntimeError(f"expected {EXPECTED_CELLS} smoke cells, got {len(selected)}")
    return selected


def _context_matches(context: dict[str, Any], row: dict[str, Any]) -> bool:
    return (
        context.get("benchmark") == row["benchmark"]
        and context.get("task") == row["task"]
        and context.get("memory_method") == row["memory_method"]
        and _model_matches(row["actor_model"], context.get("actor_model"))
        and _model_matches(row["sop_model"], context.get("sop_model"))
        and context.get("mas_framework") == row["mas_framework"]
        and str(context.get("seed")) == str(row["seed"])
        and context.get("ablation", "full") == row.get("ablation", "full")
        and str(context.get("case_id")) == str(row["case_id"])
    )


def _candidate_results(row: dict[str, Any]) -> list[Path]:
    candidates = []
    for path in RESULT_DIR.glob("sop-model-sensitivity__*.json"):
        if path.name.endswith(".team-memory-runtime-metrics.json"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        context = payload.get("context")
        if isinstance(context, dict) and _context_matches(context, row):
            candidates.append(path)
    return sorted(candidates, key=lambda item: item.stat().st_mtime)


def _latest_fresh_result(row: dict[str, Any], started_at: float) -> Path | None:
    fresh = [path for path in _candidate_results(row) if path.stat().st_mtime >= started_at - 1.0]
    return fresh[-1] if fresh else None


def _latest_existing_result(row: dict[str, Any]) -> Path | None:
    candidates = _candidate_results(row)
    return candidates[-1] if candidates else None


def _validate_result(path: Path, row: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    context = payload.get("context") or {}
    metrics = payload.get("metrics") or {}
    metadata = payload.get("metadata") or {}
    cases = payload.get("cases")
    schema: list[str] = []
    if not isinstance(context, dict) or not _context_matches(context, row):
        schema.append("context mismatch")
    if not isinstance(cases, list) or not cases:
        schema.append("missing/nonempty cases")
    if _num(metrics.get("primary_score")) is None:
        schema.append("missing metrics.primary_score")
    if _num(metrics.get("case_count")) != 1.0:
        schema.append(f"case_count={metrics.get('case_count')}")
    if "failed" in path.name or "runtime-metrics" in path.name:
        schema.append("failed/sidecar path")
    task = row["task"]
    if task == "alfworld" and metadata.get("execution_mode") != "official-alfworld-single-case-team-memory-loop":
        schema.append("ALFWorld Team Memory official execution mode missing")
    if task == "webarena" and metadata.get("execution_mode") != "official-webarena-single-case":
        schema.append("WebArena official execution mode missing")
    if task == "multiagentbench" and metadata.get("execution_mode") != "official-marble-single-case":
        schema.append("MARBLE official execution mode missing")
    if task == "officebench" and metadata.get("execution_mode") != "official-officebench-single-case":
        schema.append("OfficeBench official execution mode missing")
    if schema:
        raise RuntimeError("; ".join(schema))
    return {
        "benchmark": task,
        "mas": row["mas_framework"],
        "memory_method": row["memory_method"],
        "actor_model": row["actor_model"],
        "sop_model": row["sop_model"],
        "case_id": str(row["case_id"]),
        "official_score": float(metrics["primary_score"]),
        "sop_json_valid_rate": _num(metrics.get("sop_json_valid_rate")),
        "candidate_pass_rate": _num(metrics.get("candidate_pass_rate")),
        "sop_model_cost": _num(metrics.get("sop_model_cost")),
        "result_path": str(path.relative_to(ROOT)),
        "official_log": str(metadata.get("official_log", "")),
    }


def _write_outputs(
    rows: list[dict[str, Any]],
    plan: list[dict[str, Any]],
    preflight: list[dict[str, Any]],
    *,
    stopped: dict[str, Any] | None = None,
) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    completed = {
        (row["benchmark"], row["sop_model"], row["case_id"])
        for row in rows
    }
    missing = [
        row
        for row in plan
        if (row["task"], row["sop_model"], str(row["case_id"])) not in completed
    ]
    matched_case_issues = []
    by_model: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_model[row["sop_model"]].add(f"{row['benchmark']}:{row['case_id']}")
    if by_model:
        sets = list(by_model.values())
        if any(values != sets[0] for values in sets[1:]):
            matched_case_issues.append({model: sorted(values) for model, values in by_model.items()})
    lines = [
        "# E2 SOP-Model Sensitivity Smoke Summary",
        "",
        "This is an engineering smoke check for the E2 official-loop wiring. It is not the formal E2 result table.",
        "",
    ]
    if stopped:
        lines.extend(["## Stop Status", "", f"- `{json.dumps(stopped, ensure_ascii=False)}`", ""])
    lines.extend(
        [
            "## Smoke Results",
            "",
            "| Benchmark | MAS | Memory method | Actor model | SOP model | Case ID | Official score | Result JSON |",
            "| --- | --- | --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for row in sorted(rows, key=lambda item: (item["benchmark"], item["sop_model"])):
        lines.append(
            f"| {row['benchmark']} | {row['mas']} | {row['memory_method']} | {row['actor_model']} | {row['sop_model']} | {row['case_id']} | {row['official_score']:.3f} | {row['result_path']} |"
        )
    quality = {
        "expected_cells": len(plan),
        "completed_cells": len(rows),
        "missing_cells": [
            {
                "benchmark": row["task"],
                "sop_model": row["sop_model"],
                "case_id": str(row["case_id"]),
            }
            for row in missing
        ],
        "local_model_preflight": preflight,
        "schema_violations": [] if stopped is None or stopped.get("reason") != "result_validation_failed" else [stopped],
        "matched_case_issues": matched_case_issues,
        "stopped": stopped,
        "per_cell_result_paths": rows,
        "notes": [
            "No failed run, sidecar, or partial row is counted.",
            "Official evaluators are not modified by this smoke runner.",
        ],
    }
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    QUALITY_JSON.write_text(json.dumps(quality, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _run_cell(row: dict[str, Any], env_base: dict[str, str]) -> Path:
    tag = f"e2-smoke-{row['task']}-{row['sop_model']}-{_safe(str(row['case_id']))}-{_stable_tag(row)}"
    env = env_base.copy()
    env["TEAM_MEMORY_DB"] = str((DB_DIR / f"{tag}.team-memory.db").resolve())
    cmd = [
        sys.executable,
        "-m",
        "team_memory.evaluation_runner",
        "run",
        "--benchmark",
        row["benchmark"],
        "--task",
        row["task"],
        "--memory-method",
        row["memory_method"],
        "--actor-model",
        row["actor_model"],
        "--sop-model",
        row["sop_model"],
        "--mas",
        row["mas_framework"],
        "--seed",
        str(row["seed"]),
        "--case-id",
        str(row["case_id"]),
        "--ablation",
        row.get("ablation", "full"),
        "--state-db",
        str((STATE_DIR / f"{tag}.state.db").resolve()),
        "--no-resume",
        "--rerun-failed",
    ]
    started_at = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True)
    with RUN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n===== {task} {sop_model} {case_id} rc={rc} =====\n{out}\n{err}\n".format(
                rc=proc.returncode,
                out=proc.stdout[-20000:],
                err=proc.stderr[-20000:],
                **row,
            )
        )
    if proc.returncode != 0:
        raise RuntimeError(
            json.dumps(
                {
                    "reason": "runner_nonzero",
                    "returncode": proc.returncode,
                    "stdout_tail": proc.stdout[-2000:],
                    "stderr_tail": proc.stderr[-2000:],
                },
                ensure_ascii=False,
            )
        )
    result = _latest_fresh_result(row, started_at)
    if result is None:
        raise RuntimeError("no fresh official result JSON")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sop-models", help="Comma-separated SOP models for this smoke run.")
    parser.add_argument("--summary", help="Summary markdown path relative to repo root.")
    parser.add_argument("--quality", help="Quality JSON path relative to repo root.")
    parser.add_argument("--db-dir", help="DB directory relative to repo root.")
    parser.add_argument("--state-dir", help="State DB directory relative to repo root.")
    parser.add_argument("--run-log", help="Runner log path relative to repo root.")
    args = parser.parse_args()
    _configure_outputs(args)
    os.chdir(ROOT)
    _load_dotenv()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    RUN_LOG.write_text("", encoding="utf-8")

    preflight: list[dict[str, Any]] = []
    try:
        for model_id in SMOKE_SOP_MODELS:
            print(f"PREFLIGHT {model_id}", flush=True)
            preflight.append(_preflight_model(model_id))
    except Exception as exc:
        stopped = {"reason": "local_model_preflight_failed", "error": str(exc)}
        _write_outputs([], [], preflight, stopped=stopped)
        print(json.dumps(stopped, indent=2, ensure_ascii=False), flush=True)
        return 2

    try:
        plan = _load_smoke_plan()
    except Exception as exc:
        stopped = {"reason": "smoke_plan_failed", "error": str(exc)}
        _write_outputs([], [], preflight, stopped=stopped)
        print(json.dumps(stopped, indent=2, ensure_ascii=False), flush=True)
        return 2
    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = "src"
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(plan, start=1):
        label = f"{index}/{len(plan)} {row['task']} autogen team-memory {row['case_id']} sop={row['sop_model']}"
        try:
            existing = _latest_existing_result(row)
            if existing is not None:
                print(f"SKIP {label}", flush=True)
                result = existing
            else:
                print(f"RUN {label}", flush=True)
                result = _run_cell(row, env_base)
            checked = _validate_result(result, row)
        except Exception as exc:
            stopped = {"cell": label, "reason": "cell_failed", "error": str(exc)}
            _write_outputs(rows, plan, preflight, stopped=stopped)
            print(json.dumps(stopped, indent=2, ensure_ascii=False), flush=True)
            return 2
        rows.append(checked)
        print(f"OK {label} score={checked['official_score']}", flush=True)
        _write_outputs(rows, plan, preflight)

    _write_outputs(rows, plan, preflight)
    print(f"WROTE {SUMMARY_MD.relative_to(ROOT)}", flush=True)
    print(f"WROTE {QUALITY_JSON.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
