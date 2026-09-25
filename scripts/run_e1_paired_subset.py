#!/usr/bin/env python3
"""Run the E1 paper-runnable paired subset with stop-on-failure checks."""

from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "benchmark-results" / "plans" / "e1_paper_runnable_plan.jsonl"
RESULT_DIR = ROOT / "benchmark-results" / "unified-v3" / "results"
TABLE_DIR = ROOT / "benchmark-results" / "unified-v3" / "tables"
LOG_DIR = ROOT / "benchmark-results" / "unified-v3" / "logs"
DB_DIR = ROOT / "benchmark-results" / "unified-v3" / "e1-dbs"
STATE_DIR = ROOT / "benchmark-results" / "unified-v3" / "e1-states"
SUMMARY_MD = TABLE_DIR / "e1_cross_benchmark_generality_summary.md"
QUALITY_JSON = TABLE_DIR / "e1_cross_benchmark_generality_quality.json"
RUN_LOG = LOG_DIR / "e1_paired_subset_runner.log"

EXPECTED_CELLS = 168
ALLOWED = {
    ("alfworld", "autogen", "no-memory"),
    ("alfworld", "autogen", "gmemory"),
    ("alfworld", "autogen", "team-memory"),
    ("alfworld", "dylan", "no-memory"),
    ("alfworld", "dylan", "gmemory"),
    ("alfworld", "dylan", "team-memory"),
    ("webarena", "autogen", "no-memory"),
    ("webarena", "autogen", "team-memory"),
    ("officebench", "autogen", "no-memory"),
    ("officebench", "autogen", "team-memory"),
}


def _load_plan() -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in PLAN_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != EXPECTED_CELLS:
        raise RuntimeError(f"Expected {EXPECTED_CELLS} E1 cells, found {len(rows)} in {PLAN_PATH}")
    for row in rows:
        key = (row["task"], row["mas_framework"], row["memory_method"])
        if row.get("benchmark") != "cross-benchmark-generality":
            raise RuntimeError(f"Unexpected benchmark in E1 plan: {row}")
        if key not in ALLOWED:
            raise RuntimeError(f"Non-runnable or excluded E1 cell in plan: {key}")
        if row.get("actor_model") != "gpt-5-mini":
            raise RuntimeError(f"Unexpected actor model in E1 plan: {row}")
    return rows


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
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _context_matches(context: dict[str, Any], row: dict[str, Any]) -> bool:
    expected = {
        "benchmark": row["benchmark"],
        "task": row["task"],
        "memory_method": row["memory_method"],
        "actor_model": row["actor_model"],
        "sop_model": row["sop_model"],
        "mas_framework": row["mas_framework"],
        "seed": row["seed"],
        "ablation": row.get("ablation", "full"),
        "case_id": str(row["case_id"]),
    }
    actual = {key: context.get(key) for key in expected}
    actual["case_id"] = str(actual["case_id"])
    return actual == expected


def _candidate_results(row: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    for path in RESULT_DIR.glob("cross-benchmark-generality__*.json"):
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


def _validate_result(path: Path, row: dict[str, Any]) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    context = data.get("context") or {}
    metrics = data.get("metrics") or {}
    metadata = data.get("metadata") or {}
    cases = data.get("cases")
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
    memory_method = row["memory_method"]
    if task == "alfworld" and memory_method in {"no-memory", "gmemory"}:
        if metadata.get("adapter") != "adapters/team_memory_gmemory_adapter.py":
            schema.append("ALFWorld GMemory delegate adapter missing")
    elif task == "alfworld" and memory_method == "team-memory":
        if metadata.get("execution_mode") != "official-alfworld-single-case-team-memory-loop":
            schema.append("ALFWorld Team Memory official execution mode missing")
    elif task == "webarena":
        if metadata.get("execution_mode") != "official-webarena-single-case":
            schema.append("WebArena official execution mode missing")
    elif task == "officebench":
        if metadata.get("execution_mode") != "official-officebench-single-case":
            schema.append("OfficeBench official execution mode missing")

    if schema:
        raise RuntimeError("; ".join(schema))

    prompt_tokens = _num(metrics.get("actor_prompt_tokens")) or 0.0
    completion_tokens = _num(metrics.get("actor_completion_tokens")) or 0.0
    token_candidates = [
        _num(metrics.get("total_tokens")),
        _num(metrics.get("tokens")),
        prompt_tokens + completion_tokens if prompt_tokens or completion_tokens else None,
    ]
    latency_candidates = [
        _num(metrics.get("latency")),
        _num(metrics.get("latency_seconds")),
        _num(metrics.get("duration_seconds")),
        _num(metrics.get("runtime_seconds")),
    ]
    return {
        "benchmark": row["task"],
        "mas": row["mas_framework"],
        "memory_method": row["memory_method"],
        "actor_model": row["actor_model"],
        "sop_model": row["sop_model"],
        "case_id": str(row["case_id"]),
        "official_score": float(metrics["primary_score"]),
        "tokens": next((value for value in token_candidates if value is not None), None),
        "latency": next((value for value in latency_candidates if value is not None), None),
        "result_path": str(path.relative_to(ROOT)),
        "official_log": str(metadata.get("official_log", "")),
        "metadata": {
            "adapter": metadata.get("adapter"),
            "execution_mode": metadata.get("execution_mode"),
            "official_entrypoint": metadata.get("official_entrypoint"),
            "team_memory_injected": metadata.get("team_memory_injected"),
            "memory_method": metadata.get("memory_method"),
        },
    }


def _append_log(text: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def _run_prechecks() -> dict[str, Any]:
    env = os.environ.copy()
    url_names = [
        "WEBARENA_SHOPPING",
        "WEBARENA_SHOPPING_ADMIN",
        "WEBARENA_REDDIT",
        "WEBARENA_GITLAB",
        "WEBARENA_MAP",
        "WEBARENA_WIKIPEDIA",
    ]
    url_rows = []
    for name in url_names:
        url = env.get(name, "")
        if not url:
            url_rows.append({"name": name, "url": url, "ok": False, "status": "missing"})
            continue
        proc = subprocess.run(
            ["curl", "--noproxy", "*", "-I", "--max-time", "30", url],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        first = proc.stdout.splitlines()[0] if proc.stdout.splitlines() else ""
        ok = proc.returncode == 0 and (" 2" in first or " 3" in first)
        url_rows.append(
            {
                "name": name,
                "url": url,
                "ok": ok,
                "status": first,
                "returncode": proc.returncode,
                "stderr_tail": proc.stderr[-500:],
            }
        )
    docker = subprocess.run(
        ["docker", "info", "--format", "{{json .ServerVersion}}"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    return {
        "webarena_url_check": url_rows,
        "officebench_docker_check": {
            "ok": docker.returncode == 0,
            "returncode": docker.returncode,
            "server_version": docker.stdout.strip(),
            "stderr_tail": docker.stderr[-500:],
        },
    }


def _mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return statistics.mean(clean) if clean else None


def _bootstrap_ci(values: list[float], *, seed: int = 0, rounds: int = 2000) -> tuple[float, float] | None:
    if not values:
        return None
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    estimates = []
    for _ in range(rounds):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(statistics.mean(sample))
    estimates.sort()
    lo = estimates[int(0.025 * (rounds - 1))]
    hi = estimates[int(0.975 * (rounds - 1))]
    return lo, hi


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _matched_issues(rows: list[dict[str, Any]], plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues = []
    expected_groups: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    actual_groups: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in plan:
        expected_groups[(row["task"], row["mas_framework"])][row["memory_method"]].add(str(row["case_id"]))
    for row in rows:
        actual_groups[(row["benchmark"], row["mas"])][row["memory_method"]].add(str(row["case_id"]))
    for group, methods in expected_groups.items():
        expected_sets = list(methods.values())
        if expected_sets and any(case_set != expected_sets[0] for case_set in expected_sets[1:]):
            issues.append({"group": group, "reason": "plan_unmatched", "case_sets": {k: sorted(v) for k, v in methods.items()}})
        actual_methods = actual_groups.get(group, {})
        if actual_methods:
            actual_sets = list(actual_methods.values())
            if actual_sets and any(case_set != actual_sets[0] for case_set in actual_sets[1:]):
                issues.append({"group": group, "reason": "result_unmatched", "case_sets": {k: sorted(v) for k, v in actual_methods.items()}})
    return issues


def _write_outputs(
    rows: list[dict[str, Any]],
    plan: list[dict[str, Any]],
    *,
    prechecks: dict[str, Any],
    skipped_existing: list[str],
    stopped: dict[str, Any] | None = None,
) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    completed_keys = {
        (row["benchmark"], row["mas"], row["memory_method"], row["case_id"])
        for row in rows
    }
    missing = [
        row
        for row in plan
        if (row["task"], row["mas_framework"], row["memory_method"], str(row["case_id"])) not in completed_keys
    ]
    group_rows: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_rows[(row["benchmark"], row["mas"], row["memory_method"])].append(row)

    baseline_scores = {
        (row["benchmark"], row["mas"], row["case_id"]): row["official_score"]
        for row in rows
        if row["memory_method"] == "no-memory"
    }
    summary_rows: list[dict[str, Any]] = []
    for group in sorted(group_rows):
        values = group_rows[group]
        scores = [float(row["official_score"]) for row in values]
        deltas = [
            float(row["official_score"]) - baseline_scores[(row["benchmark"], row["mas"], row["case_id"])]
            for row in values
            if (row["benchmark"], row["mas"], row["case_id"]) in baseline_scores
        ]
        ci = _bootstrap_ci(scores, seed=int(hashlib.sha256(repr(group).encode("utf-8")).hexdigest()[:8], 16))
        summary_rows.append(
            {
                "benchmark": group[0],
                "mas": group[1],
                "memory_method": group[2],
                "cases": len({row["case_id"] for row in values}),
                "official_task_score_mean": statistics.mean(scores),
                "score_ci95": ci,
                "delta_vs_no_memory": statistics.mean(deltas) if deltas else 0.0,
                "tokens": _mean([row.get("tokens") for row in values]),
                "latency": _mean([row.get("latency") for row in values]),
                "result_rows": len(values),
            }
        )

    lines = [
        "# E1 Cross-Benchmark Generality Summary",
        "",
        "This table contains only the paper-runnable paired E1 subset. Scores are official single-case results averaged over matched case IDs within each benchmark/MAS block.",
        "",
        "The interval column is a deterministic bootstrap-over-cases 95% CI for the mean score.",
        "",
    ]
    if stopped:
        lines.extend(["## Stop Status", "", f"- Stopped: `{json.dumps(stopped, ensure_ascii=False)}`", ""])
    lines.extend(
        [
            "## Summary",
            "",
            "| Benchmark | MAS | Memory method | Cases | Official task score mean | 95% CI | Delta vs no-memory | Tokens | Latency | Result rows |",
            "| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary_rows:
        ci = row["score_ci95"]
        ci_text = "n/a" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        lines.append(
            "| {benchmark} | {mas} | {memory_method} | {cases} | {score} | {ci} | {delta} | {tokens} | {latency} | {result_rows} |".format(
                benchmark=row["benchmark"],
                mas=row["mas"],
                memory_method=row["memory_method"],
                cases=row["cases"],
                score=_fmt(row["official_task_score_mean"]),
                ci=ci_text,
                delta=_fmt(row["delta_vs_no_memory"]),
                tokens=_fmt(row["tokens"]),
                latency=_fmt(row["latency"]),
                result_rows=row["result_rows"],
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- MultiAgentBench/MARBLE is excluded from this E1 main plan until a Research-only paired matrix is explicitly adopted and smoke-tested.",
            "- AgentNet is excluded because matched no-memory/GMemory baselines are missing.",
            "- `agent-native-memory`, `generative-memory`, and `mem0` are excluded because they do not yet have real official-loop injection.",
            "- Failed runs, sidecars, partial rows, and non-plan smoke rows are not counted.",
            "",
            "## Result Paths",
            "",
            "| Benchmark | MAS | Memory method | Case ID | Official score | Result path |",
            "| --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for row in sorted(rows, key=lambda item: (item["benchmark"], item["mas"], item["memory_method"], item["case_id"])):
        lines.append(
            f"| {row['benchmark']} | {row['mas']} | {row['memory_method']} | {row['case_id']} | {row['official_score']:.3f} | {row['result_path']} |"
        )

    quality = {
        "expected_cells": len(plan),
        "completed_cells": len(rows),
        "missing_cells": [
            {
                "task": row["task"],
                "mas": row["mas_framework"],
                "memory_method": row["memory_method"],
                "case_id": str(row["case_id"]),
            }
            for row in missing
        ],
        "schema_violations": [] if stopped is None or stopped.get("reason") != "result_validation_failed" else [stopped],
        "unmatched_comparisons": _matched_issues(rows, plan),
        "excluded_cells": [
            {"scope": "MultiAgentBench/MARBLE", "reason": "no complete paired E1 cross-method matrix yet"},
            {"scope": "AgentNet", "reason": "matched no-memory/GMemory baselines missing"},
            {"scope": "agent-native-memory/generative-memory/mem0", "reason": "no real official-loop injection"},
        ],
        "skipped_existing_cells": skipped_existing,
        "stopped": stopped,
        "summary_rows": summary_rows,
        "per_cell_result_paths": rows,
        **prechecks,
    }
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    QUALITY_JSON.write_text(json.dumps(quality, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _run_cell(row: dict[str, Any], env_base: dict[str, str]) -> Path:
    tag = (
        f"{row['task']}-{row['mas_framework']}-{row['memory_method']}-"
        f"{_safe(str(row['case_id']))}-{_stable_tag(row)}"
    )
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
    _append_log(
        "\n===== {task} {mas_framework} {memory_method} {case_id} rc={rc} =====\n{out}\n{err}".format(
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
    os.chdir(ROOT)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    RUN_LOG.write_text("", encoding="utf-8")

    plan = _load_plan()
    prechecks = _run_prechecks()
    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = "src"
    rows: list[dict[str, Any]] = []
    skipped_existing: list[str] = []

    for index, row in enumerate(plan, start=1):
        label = f"{index}/{len(plan)} {row['task']} {row['mas_framework']} {row['memory_method']} {row['case_id']}"
        existing = _latest_existing_result(row)
        if existing is not None:
            try:
                checked = _validate_result(existing, row)
            except Exception as exc:
                stopped = {
                    "cell": label,
                    "reason": "existing_result_validation_failed",
                    "result_path": str(existing.relative_to(ROOT)),
                    "error": str(exc),
                }
                _write_outputs(rows, plan, prechecks=prechecks, skipped_existing=skipped_existing, stopped=stopped)
                print(json.dumps(stopped, indent=2, ensure_ascii=False), flush=True)
                return 3
            rows.append(checked)
            skipped_existing.append(label)
            print(f"SKIP {label} score={checked['official_score']}", flush=True)
            _write_outputs(rows, plan, prechecks=prechecks, skipped_existing=skipped_existing)
            continue
        try:
            print(f"RUN {label}", flush=True)
            result_path = _run_cell(row, env_base)
            checked = _validate_result(result_path, row)
        except Exception as exc:
            stopped = {"cell": label, "reason": "cell_failed", "error": str(exc)}
            _write_outputs(rows, plan, prechecks=prechecks, skipped_existing=skipped_existing, stopped=stopped)
            print(json.dumps(stopped, indent=2, ensure_ascii=False), flush=True)
            return 2
        rows.append(checked)
        print(f"OK {label} score={checked['official_score']}", flush=True)
        _write_outputs(rows, plan, prechecks=prechecks, skipped_existing=skipped_existing)

    _write_outputs(rows, plan, prechecks=prechecks, skipped_existing=skipped_existing)
    print(f"WROTE {SUMMARY_MD.relative_to(ROOT)}", flush=True)
    print(f"WROTE {QUALITY_JSON.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
