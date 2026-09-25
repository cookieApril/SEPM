#!/usr/bin/env python3
"""Run the bounded E2 SOP-model balanced mini subset."""

from __future__ import annotations

import hashlib
import json
import os
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
DB_DIR = ROOT / "benchmark-results" / "unified-v3" / "e2-mini-dbs"
STATE_DIR = ROOT / "benchmark-results" / "unified-v3" / "e2-mini-states"
SUMMARY_MD = TABLE_DIR / "e2_sop_model_mini_summary.md"
QUALITY_JSON = TABLE_DIR / "e2_sop_model_mini_quality.json"
RUN_LOG = LOG_DIR / "e2_sop_model_mini_runner.log"

MINI_CASES = {
    "alfworld": "data/alfworld/json_2.1.1/valid_unseen/pick_and_place_simple-SoapBottle-None-Toilet-424/trial_T20190907_004321_405868/game.tw-pddl",
    "webarena": "495",
    "officebench": "2-39/1",
    "multiagentbench": "research/47",
}
SOP_MODELS = [
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
EXPECTED_TARGET_CELLS = len(MINI_CASES) * len(SOP_MODELS)


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
    key_required = bool(model.get("api_key_required", True))
    api_key = os.environ.get(key_env)
    if key_required and not api_key:
        raise RuntimeError(f"{model_id}: missing API key env {key_env}")
    return {
        "model_id": model_id,
        "api_model": str(model["api_model"]),
        "family": str(model.get("family", "")),
        "base_url": str(base_url),
        "base_url_env": base_url_env,
        "api_key": api_key or "local-no-key",
        "key_env": key_env,
        "is_local": str(model.get("family", "")).startswith("local-"),
    }


def _models_endpoint(base_url: str) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/models"
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            body = response.read(4096).decode("utf-8", errors="replace")
            return {
                "ok": 200 <= response.status < 300,
                "status": response.status,
                "latency_seconds": time.perf_counter() - started,
                "body_head": body[:1000],
                "url": url,
            }
    except (urllib.error.URLError, TimeoutError) as exc:
        return {
            "ok": False,
            "status": None,
            "latency_seconds": time.perf_counter() - started,
            "error": repr(exc),
            "url": url,
        }


def _preflight_model(model_id: str) -> dict[str, Any]:
    connection = _model_connection(model_id)
    model = _model_config(model_id)
    endpoint = _models_endpoint(connection["base_url"]) if connection["is_local"] else None
    if endpoint is not None and not endpoint["ok"]:
        return {**connection, "ok": False, "models_endpoint": endpoint, "error": "local /v1/models unavailable"}
    client = OpenAI(
        api_key=str(connection["api_key"]),
        base_url=str(connection["base_url"]),
        timeout=120,
        max_retries=0,
    )
    prompt = (
        "Return only valid compact JSON for one conservative SOP candidate with keys "
        "title, applicability, exclusions, atomic_steps, safety_constraints, and cost_estimate."
    )
    kwargs: dict[str, Any] = {
        "model": connection["api_model"],
        "messages": [
            {"role": "system", "content": "You emit strict JSON only."},
            {"role": "user", "content": prompt},
        ],
    }
    if str(connection["api_model"]).startswith("gpt-5"):
        kwargs["max_completion_tokens"] = 1024
    else:
        kwargs["temperature"] = float(model.get("temperature", 0.0))
        kwargs["max_tokens"] = min(int(model.get("max_tokens", 1024)), 1024)
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
        parsed = json.loads(text)
        required = {
            "title",
            "applicability",
            "exclusions",
            "atomic_steps",
            "safety_constraints",
            "cost_estimate",
        }
        missing = sorted(required - set(parsed))
        if missing:
            raise RuntimeError(f"JSON missing keys {missing}")
        usage = getattr(response, "usage", None)
        return {
            **connection,
            "ok": True,
            "models_endpoint": endpoint,
            "json_parse_ok": True,
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
            **connection,
            "ok": False,
            "models_endpoint": endpoint,
            "json_parse_ok": False,
            "latency_seconds": time.perf_counter() - started,
            "error": str(exc),
        }


def _load_mini_plan(available_models: set[str]) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in PLAN_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = []
    for model in SOP_MODELS:
        if model not in available_models:
            continue
        for task, case_id in MINI_CASES.items():
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
                raise RuntimeError(f"expected at most one mini cell for {task} {case_id} {model}, got {len(matches)}")
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
    task = row["task"]
    modes = {
        "alfworld": "official-alfworld-single-case-team-memory-loop",
        "webarena": "official-webarena-single-case",
        "multiagentbench": "official-marble-single-case",
        "officebench": "official-officebench-single-case",
    }
    if metadata.get("execution_mode") != modes[task]:
        schema.append(f"{task} official execution mode missing")
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
) -> None:
    completed = {(row["benchmark"], row["sop_model"], row["case_id"]) for row in rows}
    missing = [
        row
        for row in plan
        if (row["task"], row["sop_model"], str(row["case_id"])) not in completed
    ]
    unavailable = [row for row in preflight if not row.get("ok")]
    score_sources = Counter(row["score_source"] for row in rows)
    matched_case_issues = []
    by_model: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_model[row["sop_model"]].add(f"{row['benchmark']}:{row['case_id']}")
    if by_model:
        expected = {f"{task}:{case_id}" for task, case_id in MINI_CASES.items()}
        for model, cases in by_model.items():
            if cases != expected:
                matched_case_issues.append({"sop_model": model, "cases": sorted(cases), "expected": sorted(expected)})

    lines = [
        "# E2 SOP-Model Sensitivity Mini Summary",
        "",
        "This is a balanced mini subset for E2 wiring and SOP-model sanity checks. It is not the full 864-cell E2 table.",
        "",
    ]
    if stopped:
        lines.extend(["## Stop Status", "", f"- `{json.dumps(stopped, ensure_ascii=False)}`", ""])
    lines.extend(
        [
            "## Mini Results",
            "",
            "| SOP model | Benchmark | Case ID | Official score | Score source | SOP parse success | SOP reuse count | SOP reuse success | Token usage | Latency | Result path |",
            "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in sorted(rows, key=lambda item: (item["sop_model"], item["benchmark"])):
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
    means = []
    by_sop: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_sop[row["sop_model"]].append(row["official_score"])
    for model in SOP_MODELS:
        scores = by_sop.get(model, [])
        if scores:
            means.append({"sop_model": model, "completed_cases": len(scores), "mean_score": sum(scores) / len(scores)})
    quality = {
        "target_cells": EXPECTED_TARGET_CELLS,
        "expected_cells": EXPECTED_TARGET_CELLS,
        "planned_available_cells": len(plan),
        "completed_cells": len(rows),
        "unavailable_models": unavailable,
        "skipped_unavailable": [
            {"sop_model": item["model_id"], "skipped_cells": len(MINI_CASES), "error": item.get("error", "")}
            for item in unavailable
        ],
        "missing_cells": [
            {"benchmark": row["task"], "sop_model": row["sop_model"], "case_id": str(row["case_id"])}
            for row in missing
        ],
        "schema_violations": [] if stopped is None or stopped.get("reason") != "result_validation_failed" else [stopped],
        "matched_case_issues": matched_case_issues,
        "runtime_failures": [] if stopped is None or stopped.get("reason") == "result_validation_failed" else [stopped],
        "score_source_distribution": dict(score_sources),
        "model_mean_scores": means,
        "per_cell_result_paths": rows,
        "preflight": preflight,
        "stopped": stopped,
        "notes": [
            "Official scored 0.0 rows are accepted results.",
            "No failed run, sidecar, or partial row is counted.",
            "Official evaluator scoring code is not modified by this runner.",
        ],
    }
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    QUALITY_JSON.write_text(json.dumps(quality, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _run_cell(row: dict[str, Any], env_base: dict[str, str]) -> Path:
    tag = f"e2-mini-{row['task']}-{row['sop_model']}-{_safe(str(row['case_id']))}-{_stable_tag(row)}"
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
    os.chdir(ROOT)
    _load_dotenv()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    RUN_LOG.write_text("", encoding="utf-8")

    preflight = []
    for model_id in SOP_MODELS:
        print(f"PREFLIGHT {model_id}", flush=True)
        preflight.append(_preflight_model(model_id))
        status = "OK" if preflight[-1].get("ok") else "UNAVAILABLE"
        print(f"{status} {model_id}", flush=True)
    available = {row["model_id"] for row in preflight if row.get("ok")}
    plan = _load_mini_plan(available)
    rows: list[dict[str, Any]] = []
    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = "src"

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
    print(json.dumps({"completed": len(rows), "planned_available": len(plan), "target": EXPECTED_TARGET_CELLS}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
