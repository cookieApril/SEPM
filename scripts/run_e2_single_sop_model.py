#!/usr/bin/env python3
"""Run the E2 formal subset for one SOP model only."""

from __future__ import annotations

import hashlib
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "benchmark-results" / "plans" / "sop-model-sensitivity.jsonl"
RESULT_DIR = ROOT / "benchmark-results" / "unified-v3" / "results"
TABLE_DIR = ROOT / "benchmark-results" / "unified-v3" / "tables"
LOG_DIR = ROOT / "benchmark-results" / "unified-v3" / "logs"
DB_DIR = ROOT / "benchmark-results" / "unified-v3" / "e2-local-deployed-dbs"
STATE_DIR = ROOT / "benchmark-results" / "unified-v3" / "e2-local-deployed-states"
LOCAL_SUMMARY_MD = TABLE_DIR / "e2_local_deployed_models_summary.md"
LOCAL_QUALITY_JSON = TABLE_DIR / "e2_local_deployed_models_quality.json"
AVAILABLE_SUMMARY_MD = TABLE_DIR / "e2_available_sop_models_summary.md"
AVAILABLE_QUALITY_JSON = TABLE_DIR / "e2_available_sop_models_quality.json"
GEMMA_SUMMARY_MD = TABLE_DIR / "e2_gemma_models_summary.md"
GEMMA_QUALITY_JSON = TABLE_DIR / "e2_gemma_models_quality.json"
JSON_RESPONSE_FORMAT_MODELS = {
    "qwen3.5-9b",
    "qwen3.5-27b",
    "gemma-4-12b-it",
    "gemma-4-31b-it",
    "deepseek-v4-flash-0731",
    "glm-5.2",
    "kimi-k3",
}
LOCAL_DEPLOYED_MODEL_PREFIXES = ("qwen3.5-", "gemma-4-")

EXPECTED_CASES_BY_TASK = {
    "alfworld": 12,
    "webarena": 24,
    "multiagentbench": 12,
    "officebench": 24,
}
DEFAULT_SOP_MODEL = "qwen3.5-27b"
SOP_MODELS: list[str] = [DEFAULT_SOP_MODEL]
EXPECTED_CELLS = sum(EXPECTED_CASES_BY_TASK.values())
SUMMARY_MD = TABLE_DIR / f"e2_{DEFAULT_SOP_MODEL}_summary.md"
QUALITY_JSON = TABLE_DIR / f"e2_{DEFAULT_SOP_MODEL}_quality.json"
RUN_LOG = LOG_DIR / f"e2_{DEFAULT_SOP_MODEL}_runner.log"


def _configure_model_outputs(sop_model: str) -> None:
    global SOP_MODELS, EXPECTED_CELLS, SUMMARY_MD, QUALITY_JSON, RUN_LOG
    SOP_MODELS = [sop_model]
    EXPECTED_CELLS = sum(EXPECTED_CASES_BY_TASK.values())
    SUMMARY_MD = TABLE_DIR / f"e2_{sop_model}_summary.md"
    QUALITY_JSON = TABLE_DIR / f"e2_{sop_model}_quality.json"
    RUN_LOG = LOG_DIR / f"e2_{sop_model}_runner.log"


def _load_dotenv() -> None:
    source = ROOT / ".env"
    if not source.exists():
        return
    for raw in source.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _matrix() -> dict[str, Any]:
    return json.loads((ROOT / "evaluation_matrix.json").read_text(encoding="utf-8"))


def _model_config(model_id: str) -> dict[str, Any]:
    return next(model for model in _matrix()["models"] if model["id"] == model_id)


def _model_matches(expected_id: str, observed: Any) -> bool:
    if observed is None:
        return False
    observed_text = str(observed)
    if observed_text == expected_id:
        return True
    return observed_text == str(_model_config(expected_id)["api_model"])


def _model_connection(model_id: str) -> dict[str, Any]:
    config = _matrix()
    provider = config["provider"]
    model = _model_config(model_id)
    base_url_env = str(model.get("base_url_env", provider["base_url_env"]))
    key_env = str(model.get("key_env", provider["key_env"]))
    base_url = os.environ.get(base_url_env, model.get("base_url_default"))
    if not base_url and base_url_env == provider["base_url_env"]:
        base_url = provider.get("base_url_default")
    if not base_url:
        raise RuntimeError(f"{model_id}: missing base URL from {base_url_env}")
    api_key = os.environ.get(key_env) or "local-no-key"
    return {
        "model_id": model_id,
        "api_model": str(model["api_model"]),
        "base_url": str(base_url),
        "api_key": api_key,
    }


def _public_connection(connection: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": connection["model_id"],
        "api_model": connection["api_model"],
        "base_url": connection["base_url"],
        "api_key_present": bool(connection.get("api_key")),
    }


def _models_endpoint(base_url: str, api_key: str | None = None) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/models"
    started = time.perf_counter()
    try:
        request = urllib.request.Request(url)
        if api_key:
            request.add_header("Authorization", f"Bearer {api_key}")
        with urllib.request.urlopen(request, timeout=20) as response:
            body = response.read(4096).decode("utf-8", errors="replace")
            return {
                "ok": 200 <= response.status < 300,
                "status": response.status,
                "latency_seconds": time.perf_counter() - started,
                "body_head": body[:1000],
                "url": url,
            }
    except (urllib.error.URLError, TimeoutError) as exc:
        if api_key:
            try:
                client = OpenAI(api_key=api_key, base_url=base_url, timeout=20, max_retries=0)
                models = client.models.list()
                ids = [model.id for model in models.data]
                return {
                    "ok": True,
                    "status": "sdk-fallback",
                    "latency_seconds": time.perf_counter() - started,
                    "body_head": json.dumps({"ids": ids[:20]}),
                    "url": url,
                    "urllib_error": repr(exc),
                }
            except Exception:
                pass
        return {
            "ok": False,
            "status": None,
            "latency_seconds": time.perf_counter() - started,
            "error": repr(exc),
            "url": url,
        }


def _preflight_model(model_id: str) -> dict[str, Any]:
    connection = _model_connection(model_id)
    public_connection = _public_connection(connection)
    endpoint = _models_endpoint(connection["base_url"], str(connection.get("api_key") or ""))
    if not endpoint["ok"]:
        return {**public_connection, "ok": False, "models_endpoint": endpoint, "error": "/v1/models unavailable"}
    client = OpenAI(
        api_key=str(connection["api_key"]),
        base_url=str(connection["base_url"]),
        timeout=120,
        max_retries=0,
    )
    started = time.perf_counter()
    attempts = [
        (
            "Return only valid compact JSON for one conservative SOP candidate with keys "
            "title, applicability, exclusions, atomic_steps, safety_constraints, and cost_estimate."
        ),
        (
            "Return exactly one JSON object with all of these keys: "
            '{"title":"check","applicability":["single benchmark case"],'
            '"exclusions":["none"],"atomic_steps":["observe","act","verify"],'
            '"safety_constraints":["preserve authorization and confirmation"],'
            '"cost_estimate":{"tokens":0,"latency_seconds":0}}'
        ),
    ]
    try:
        required = {
            "title",
            "applicability",
            "exclusions",
            "atomic_steps",
            "safety_constraints",
            "cost_estimate",
        }
        response = None
        missing: list[str] = []
        for prompt in attempts:
            kwargs: dict[str, Any] = {
                "model": connection["api_model"],
                "messages": [
                    {"role": "system", "content": "You emit strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 1024,
            }
            if model_id in JSON_RESPONSE_FORMAT_MODELS:
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content or ""
            parsed = _parse_jsonish_object(text)
            missing = sorted(required - set(parsed))
            if not missing:
                break
        else:
            raise RuntimeError(f"JSON missing keys {missing}")
        assert response is not None
        usage = getattr(response, "usage", None)
        return {
            **public_connection,
            "ok": True,
            "models_endpoint": endpoint,
            "json_parse_ok": True,
            "response_format_json_object": model_id in JSON_RESPONSE_FORMAT_MODELS,
            "latency_seconds": time.perf_counter() - started,
            "usage": {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            },
            "cost_accounting_ok": usage is not None,
        }
    except Exception as exc:
        return {
            **public_connection,
            "ok": False,
            "models_endpoint": endpoint,
            "json_parse_ok": False,
            "latency_seconds": time.perf_counter() - started,
            "error": str(exc),
        }


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


def _parse_jsonish_object(text: str) -> dict[str, Any]:
    stripped = (text or "").strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL | re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1).strip()
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise TypeError("SOP model response must be a JSON object")
    return value


def _load_plan(sop_model: str) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in PLAN_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [
        row
        for row in rows
        if row["benchmark"] == "sop-model-sensitivity"
        and row["task"] in EXPECTED_CASES_BY_TASK
        and row["sop_model"] == sop_model
        and row["mas_framework"] == "autogen"
        and row["actor_model"] == "gpt-5-mini"
        and row["memory_method"] == "team-memory"
        and row.get("ablation", "full") == "full"
        and row["seed"] == 0
    ]
    counts: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in selected:
        counts[(row["sop_model"], row["task"])].add(str(row["case_id"]))
    issues = []
    for model in SOP_MODELS:
        for task, expected in EXPECTED_CASES_BY_TASK.items():
            got = len(counts[(model, task)])
            if got != expected:
                issues.append({"sop_model": model, "task": task, "expected": expected, "got": got})
    if issues:
        raise RuntimeError(f"matched E2 local plan issue: {json.dumps(issues, ensure_ascii=False)}")
    if len(selected) != EXPECTED_CELLS:
        raise RuntimeError(f"expected {EXPECTED_CELLS} local cells, got {len(selected)}")
    task_order = {task: index for index, task in enumerate(EXPECTED_CASES_BY_TASK)}
    return sorted(selected, key=lambda row: (task_order[row["task"]], str(row["case_id"])))


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


def _latest_existing_result(row: dict[str, Any]) -> Path | None:
    candidates = _candidate_results(row)
    return candidates[-1] if candidates else None


def _latest_fresh_result(row: dict[str, Any], started_at: float) -> Path | None:
    fresh = [path for path in _candidate_results(row) if path.stat().st_mtime >= started_at - 1.0]
    return fresh[-1] if fresh else None


def _result_token_usage(metrics: dict[str, Any]) -> float | None:
    candidates = (
        metrics.get("token_usage"),
        metrics.get("total_tokens"),
        metrics.get("marble_token_usage"),
        metrics.get("tokens"),
    )
    return next((value for value in (_num(item) for item in candidates) if value is not None), None)


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
    modes = {
        "alfworld": "official-alfworld-single-case-team-memory-loop",
        "webarena": "official-webarena-single-case",
        "multiagentbench": "official-marble-single-case",
        "officebench": "official-officebench-single-case",
    }
    if metadata.get("execution_mode") != modes[row["task"]]:
        schema.append(f"{row['task']} official execution mode missing")
    if schema:
        raise RuntimeError("; ".join(schema))
    return {
        "benchmark": row["task"],
        "mas": row["mas_framework"],
        "memory_method": row["memory_method"],
        "actor_model": row["actor_model"],
        "sop_model": row["sop_model"],
        "case_id": str(row["case_id"]),
        "official_score": float(metrics["primary_score"]),
        "score_source": str(metadata.get("official_score_source") or "official_runtime"),
        "sop_parse_success": _num(metrics.get("sop_json_valid_rate")),
        "sop_reuse_count": _num(metrics.get("sop_retrieval_count")),
        "sop_reuse_success": _num(metrics.get("sop_reuse_success")),
        "token_usage": _result_token_usage(metrics),
        "latency": _num(metrics.get("latency_seconds") or metrics.get("latency") or metrics.get("duration_seconds")),
        "result_path": str(path.relative_to(ROOT)),
        "official_log": str(metadata.get("official_log", "")),
    }


def _format_num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _write_outputs(
    rows: list[dict[str, Any]],
    plan: list[dict[str, Any]],
    preflight: list[dict[str, Any]],
    *,
    stopped: dict[str, Any] | None = None,
    skipped_existing: int = 0,
) -> None:
    completed_keys = {(row["benchmark"], row["sop_model"], row["case_id"]) for row in rows}
    missing = [
        row
        for row in plan
        if (row["task"], row["sop_model"], str(row["case_id"])) not in completed_keys
    ]
    score_sources = Counter(row["score_source"] for row in rows)
    by_model_task: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_model: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_model_task[(row["sop_model"], row["benchmark"])].append(row["official_score"])
        by_model[row["sop_model"]].append(row["official_score"])
    model_task_means = [
        {
            "sop_model": model,
            "benchmark": task,
            "cases": len(by_model_task[(model, task)]),
            "mean_score": sum(by_model_task[(model, task)]) / len(by_model_task[(model, task)]),
        }
        for model in SOP_MODELS
        for task in EXPECTED_CASES_BY_TASK
        if by_model_task[(model, task)]
    ]
    model_means = [
        {
            "sop_model": model,
            "cases": len(by_model[model]),
            "overall_mean_score": sum(by_model[model]) / len(by_model[model]),
        }
        for model in SOP_MODELS
        if by_model[model]
    ]
    matched_case_issues = []
    for task, expected_count in EXPECTED_CASES_BY_TASK.items():
        expected_sets = {
            model: {str(row["case_id"]) for row in plan if row["sop_model"] == model and row["task"] == task}
            for model in SOP_MODELS
        }
        first = expected_sets[SOP_MODELS[0]]
        for model, values in expected_sets.items():
            if len(values) != expected_count or values != first:
                matched_case_issues.append(
                    {"benchmark": task, "sop_model": model, "case_count": len(values), "expected_count": expected_count}
                )

    lines = [
        "# E2 Local Deployed Models Summary",
        "",
        "This table covers only the already deployed local SOP models. It is not the full 864-cell E2 table.",
        "",
    ]
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
                parse=_format_num(row["sop_parse_success"]),
                reuse_count=_format_num(row["sop_reuse_count"]),
                reuse_success=_format_num(row["sop_reuse_success"]),
                tokens=_format_num(row["token_usage"]),
                latency=_format_num(row["latency"]),
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
        "expected_cells": EXPECTED_CELLS,
        "completed_cells": len(rows),
        "skipped_existing_valid_results": skipped_existing,
        "missing_cells": [
            {"benchmark": row["task"], "sop_model": row["sop_model"], "case_id": str(row["case_id"])}
            for row in missing
        ],
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
            "Official scored 0.0 rows are accepted results.",
            "No failed run, sidecar, or partial row is counted.",
            "Official evaluator scoring code is not modified by this runner.",
            "Execution is serial to avoid concurrent requests to the same local endpoint and shared external services.",
        ],
    }
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    QUALITY_JSON.write_text(json.dumps(quality, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _single_model_quality_paths(prefixes: tuple[str, ...] | None = None) -> list[Path]:
    excluded = {
        LOCAL_QUALITY_JSON.name,
        AVAILABLE_QUALITY_JSON.name,
        GEMMA_QUALITY_JSON.name,
        "e2_smoke_quality.json",
        "e2_gemma_smoke_quality.json",
        "e2_sop_model_mini_quality.json",
        "e2_readiness_report.json",
        "e2_unavailable_model_diagnosis.json",
    }
    paths: list[Path] = []
    for path in sorted(TABLE_DIR.glob("e2_*_quality.json")):
        if path.name in excluded:
            continue
        model_name = path.name.removeprefix("e2_").removesuffix("_quality.json")
        if prefixes is not None and not model_name.startswith(prefixes):
            continue
        paths.append(path)
    return paths


def _write_merged_outputs(
    summary_path: Path,
    quality_path: Path,
    quality_paths: list[Path],
    *,
    title: str,
    description: str,
    notes: list[str],
) -> None:
    rows: list[dict[str, Any]] = []
    seen_rows: set[tuple[str, str, str]] = set()
    expected_row_keys: set[tuple[str, str, str]] = set()
    preflight: list[dict[str, Any]] = []
    stopped: list[dict[str, Any]] = []
    skipped_existing = 0
    expected_cells = 0
    schema_violations: list[dict[str, Any]] = []
    matched_case_issues: list[dict[str, Any]] = []
    runtime_failures: list[dict[str, Any]] = []
    missing_cells: list[dict[str, Any]] = []
    for path in quality_paths:
        try:
            quality = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        expected_cells += int(quality.get("expected_cells") or 0)
        skipped_existing += int(quality.get("skipped_existing_valid_results") or 0)
        for row in quality.get("per_cell_result_paths") or []:
            key = (str(row.get("sop_model")), str(row.get("benchmark")), str(row.get("case_id")))
            expected_row_keys.add(key)
            if key not in seen_rows:
                rows.append(row)
                seen_rows.add(key)
        preflight.extend(quality.get("preflight") or [])
        schema_violations.extend(quality.get("schema_violations") or [])
        matched_case_issues.extend(quality.get("matched_case_issues") or [])
        runtime_failures.extend(quality.get("runtime_failures") or [])
        for missing in quality.get("missing_cells") or []:
            key = (
                str(missing.get("sop_model")),
                str(missing.get("benchmark") or missing.get("task")),
                str(missing.get("case_id")),
            )
            expected_row_keys.add(key)
            missing_cells.append(missing)
        if quality.get("stopped"):
            stopped.append(quality["stopped"])
    if expected_row_keys:
        expected_cells = len(expected_row_keys)

    score_sources = Counter(row["score_source"] for row in rows)
    by_model_task: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_model: dict[str, list[float]] = defaultdict(list)
    models = sorted({row["sop_model"] for row in rows})
    for row in rows:
        by_model_task[(row["sop_model"], row["benchmark"])].append(row["official_score"])
        by_model[row["sop_model"]].append(row["official_score"])
    model_task_means = [
        {
            "sop_model": model,
            "benchmark": task,
            "cases": len(by_model_task[(model, task)]),
            "mean_score": sum(by_model_task[(model, task)]) / len(by_model_task[(model, task)]),
        }
        for model in models
        for task in EXPECTED_CASES_BY_TASK
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
    lines = [
        f"# {title}",
        "",
        description,
        "",
        "## Per-Cell Results",
        "",
        "| SOP model | Benchmark | Case ID | Official score | Score source | SOP parse success | SOP reuse count | SOP reuse success | Token usage | Latency | Result path |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(rows, key=lambda item: (item["sop_model"], item["benchmark"], item["case_id"])):
        lines.append(
            "| {sop_model} | {benchmark} | {case_id} | {score:.3f} | {source} | {parse} | {reuse_count} | {reuse_success} | {tokens} | {latency} | {path} |".format(
                sop_model=row["sop_model"],
                benchmark=row["benchmark"],
                case_id=row["case_id"],
                score=row["official_score"],
                source=row["score_source"],
                parse=_format_num(row["sop_parse_success"]),
                reuse_count=_format_num(row["sop_reuse_count"]),
                reuse_success=_format_num(row["sop_reuse_success"]),
                tokens=_format_num(row["token_usage"]),
                latency=_format_num(row["latency"]),
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
        "expected_cells": expected_cells,
        "completed_cells": len(rows),
        "skipped_existing_valid_results": skipped_existing,
        "missing_cells": missing_cells,
        "schema_violations": schema_violations,
        "matched_case_issues": matched_case_issues,
        "runtime_failures": runtime_failures,
        "score_source_distribution": dict(score_sources),
        "model_benchmark_mean_scores": model_task_means,
        "model_overall_mean_scores": model_means,
        "per_cell_result_paths": rows,
        "preflight": preflight,
        "stopped": stopped or None,
        "notes": notes,
    }
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    quality_path.write_text(json.dumps(quality, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_local_merged_outputs() -> None:
    _write_merged_outputs(
        LOCAL_SUMMARY_MD,
        LOCAL_QUALITY_JSON,
        _single_model_quality_paths(LOCAL_DEPLOYED_MODEL_PREFIXES),
        title="E2 Local Deployed Models Summary",
        description="This table covers local SOP models that have completed formal E2 subsets so far. It is not the full 864-cell E2 table.",
        notes=[
            "Merged from completed single-local-model E2 quality files.",
            "Official scored 0.0 rows are accepted results.",
            "No failed run, sidecar, or partial row is counted.",
            "Official evaluator scoring code is not modified by these runners.",
        ],
    )


def _write_gemma_merged_outputs() -> None:
    _write_merged_outputs(
        GEMMA_SUMMARY_MD,
        GEMMA_QUALITY_JSON,
        _single_model_quality_paths(("gemma-4-",)),
        title="E2 Gemma Models Summary",
        description="This table covers Gemma local SOP models that have completed formal E2 subsets so far. It is not the full 864-cell E2 table.",
        notes=[
            "Merged from completed single-Gemma-model E2 quality files.",
            "Official scored 0.0 rows are accepted results.",
            "No failed run, sidecar, or partial row is counted.",
            "Official evaluator scoring code is not modified by these runners.",
        ],
    )


def _write_available_merged_outputs() -> None:
    quality_paths = _single_model_quality_paths()
    rows: list[dict[str, Any]] = []
    seen_rows: set[tuple[str, str, str]] = set()
    expected_row_keys: set[tuple[str, str, str]] = set()
    preflight: list[dict[str, Any]] = []
    stopped: list[dict[str, Any]] = []
    skipped_existing = 0
    expected_cells = 0
    schema_violations: list[dict[str, Any]] = []
    matched_case_issues: list[dict[str, Any]] = []
    runtime_failures: list[dict[str, Any]] = []
    missing_cells: list[dict[str, Any]] = []
    for path in quality_paths:
        try:
            quality = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        expected_cells += int(quality.get("expected_cells") or 0)
        skipped_existing += int(quality.get("skipped_existing_valid_results") or 0)
        for row in quality.get("per_cell_result_paths") or []:
            key = (str(row.get("sop_model")), str(row.get("benchmark")), str(row.get("case_id")))
            expected_row_keys.add(key)
            if key not in seen_rows:
                rows.append(row)
                seen_rows.add(key)
        preflight.extend(quality.get("preflight") or [])
        schema_violations.extend(quality.get("schema_violations") or [])
        matched_case_issues.extend(quality.get("matched_case_issues") or [])
        runtime_failures.extend(quality.get("runtime_failures") or [])
        for missing in quality.get("missing_cells") or []:
            key = (
                str(missing.get("sop_model")),
                str(missing.get("benchmark") or missing.get("task")),
                str(missing.get("case_id")),
            )
            expected_row_keys.add(key)
            missing_cells.append(missing)
        if quality.get("stopped"):
            stopped.append(quality["stopped"])
    if expected_row_keys:
        expected_cells = len(expected_row_keys)

    score_sources = Counter(row["score_source"] for row in rows)
    by_model_task: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_model: dict[str, list[float]] = defaultdict(list)
    models = sorted({row["sop_model"] for row in rows})
    for row in rows:
        by_model_task[(row["sop_model"], row["benchmark"])].append(row["official_score"])
        by_model[row["sop_model"]].append(row["official_score"])
    model_task_means = [
        {
            "sop_model": model,
            "benchmark": task,
            "cases": len(by_model_task[(model, task)]),
            "mean_score": sum(by_model_task[(model, task)]) / len(by_model_task[(model, task)]),
        }
        for model in models
        for task in EXPECTED_CASES_BY_TASK
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
    lines = [
        "# E2 Available SOP Models Summary",
        "",
        "This table merges completed formal E2 subsets for currently available SOP models. It is not the full 864-cell E2 table.",
        "",
        "## Means",
        "",
        "| SOP model | Benchmark | Cases | Mean official score |",
        "| --- | --- | ---: | ---: |",
    ]
    for item in model_task_means:
        lines.append(f"| {item['sop_model']} | {item['benchmark']} | {item['cases']} | {item['mean_score']:.3f} |")
    lines.extend(["", "| SOP model | Cases | Overall mean score |", "| --- | ---: | ---: |"])
    for item in model_means:
        lines.append(f"| {item['sop_model']} | {item['cases']} | {item['overall_mean_score']:.3f} |")
    quality = {
        "expected_cells": expected_cells,
        "completed_cells": len(rows),
        "skipped_existing_valid_results": skipped_existing,
        "missing_cells": missing_cells,
        "schema_violations": schema_violations,
        "matched_case_issues": matched_case_issues,
        "runtime_failures": runtime_failures,
        "score_source_distribution": dict(score_sources),
        "model_benchmark_mean_scores": model_task_means,
        "model_overall_mean_scores": model_means,
        "per_cell_result_paths": rows,
        "preflight": preflight,
        "stopped": stopped or None,
        "notes": [
            "Relay/API SOP models are not labeled as local models.",
            "Official scored 0.0 rows are accepted results.",
            "No failed run, sidecar, or partial row is counted.",
            "Official evaluator scoring code is not modified by these runners.",
        ],
    }
    AVAILABLE_SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    AVAILABLE_QUALITY_JSON.write_text(json.dumps(quality, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _run_cell(row: dict[str, Any], env_base: dict[str, str]) -> Path:
    tag = f"e2-local-{row['task']}-{row['sop_model']}-{_safe(str(row['case_id']))}-{_stable_tag(row)}"
    env = env_base.copy()
    env["TEAM_MEMORY_DB"] = str((DB_DIR / f"{tag}.team-memory.db").resolve())
    env.setdefault("TEAM_MEMORY_MARBLE_TIMEOUT_SECONDS", "7200")
    env.setdefault("TEAM_MEMORY_MARBLE_LLM_TIMEOUT_SECONDS", "120")
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
                    "stdout_tail": proc.stdout[-3000:],
                    "stderr_tail": proc.stderr[-3000:],
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
    parser.add_argument("--sop-model", default=DEFAULT_SOP_MODEL)
    args = parser.parse_args()
    sop_model = args.sop_model
    _configure_model_outputs(sop_model)
    os.chdir(ROOT)
    _load_dotenv()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    RUN_LOG.write_text("", encoding="utf-8")

    preflight = []
    for model_id in [sop_model]:
        print(f"PREFLIGHT {model_id}", flush=True)
        checked = _preflight_model(model_id)
        preflight.append(checked)
        status = "OK" if checked.get("ok") else "UNAVAILABLE"
        print(f"{status} {model_id}", flush=True)
        if not checked.get("ok"):
            stopped = {"cell": f"preflight {model_id}", "reason": "local_endpoint_failure", "error": checked.get("error")}
            _write_outputs([], [], preflight, stopped=stopped)
            return 2

    plan = _load_plan(sop_model)
    rows: list[dict[str, Any]] = []
    skipped_existing = 0
    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = "src"

    for index, row in enumerate(plan, start=1):
        label = f"{index}/{len(plan)} {row['task']} autogen team-memory {row['case_id']} sop={row['sop_model']}"
        try:
            existing = _latest_existing_result(row)
            if existing is not None:
                checked = _validate_result(existing, row)
                skipped_existing += 1
                print(f"SKIP {label} score={checked['official_score']}", flush=True)
            else:
                print(f"RUN {label}", flush=True)
                result = _run_cell(row, env_base)
                checked = _validate_result(result, row)
            rows.append(checked)
            _write_outputs(rows, plan, preflight, skipped_existing=skipped_existing)
            print(f"OK {label} score={checked['official_score']}", flush=True)
        except Exception as exc:
            stopped = {"cell": label, "reason": "cell_failed", "error": str(exc)}
            _write_outputs(rows, plan, preflight, stopped=stopped, skipped_existing=skipped_existing)
            print(json.dumps(stopped, indent=2, ensure_ascii=False), flush=True)
            return 2

    _write_outputs(rows, plan, preflight, skipped_existing=skipped_existing)
    if sop_model.startswith(LOCAL_DEPLOYED_MODEL_PREFIXES):
        _write_local_merged_outputs()
    if sop_model.startswith("gemma-4-"):
        _write_gemma_merged_outputs()
    _write_available_merged_outputs()
    print(json.dumps({"completed": len(rows), "expected": EXPECTED_CELLS, "skipped_existing": skipped_existing}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
