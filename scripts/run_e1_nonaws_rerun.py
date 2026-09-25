#!/usr/bin/env python3
"""Run the fresh non-AWS E1/Table 1 rerun.

This runner deliberately does not use the historical 168-cell checkpoint.  It
creates a new run directory, keeps WebArena as infrastructure-deferred, and
executes only official single-case adapter commands for the requested non-AWS
benchmarks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "benchmark-results" / "unified-v3" / "tables"
RUNS_ROOT = ROOT / "benchmark-results" / "e1-nonaws-rerun"
HOSTS = ("autogen", "dylan")
MEMORY_METHODS = ("no-memory", "gmemory", "team-memory")
WEB_ARENA_DEFER_REASON = "aws_excluded_by_user"


@dataclass(frozen=True)
class Cell:
    benchmark: str
    scenario: str
    case_id: str
    host: str
    memory_method: str
    actor_model: str = "gpt-5-mini"
    sop_model: str = "gpt-5-mini"
    seed: int = 0
    ablation: str = "full"

    @property
    def task(self) -> str:
        return self.benchmark

    @property
    def effective_sop_model(self) -> str:
        return self.sop_model if self.memory_method == "team-memory" else "not-applicable"

    @property
    def identity(self) -> str:
        payload = {
            "benchmark": "cross-benchmark-generality",
            "task": self.task,
            "scenario": self.scenario,
            "case_id": self.case_id,
            "host": self.host,
            "memory_method": self.memory_method,
            "actor_model": self.actor_model,
            "sop_model": self.effective_sop_model,
            "seed": self.seed,
            "ablation": self.ablation,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @property
    def key(self) -> str:
        readable = "__".join(
            _safe(part)
            for part in (
                "cross-benchmark-generality",
                self.task,
                self.scenario,
                self.memory_method,
                self.actor_model,
                self.effective_sop_model,
                self.host,
                self.ablation,
                f"s{self.seed}",
                self.case_id,
            )
        )
        digest = hashlib.sha256(self.identity.encode("utf-8")).hexdigest()[:12]
        return f"{readable[:180].rstrip('-_.')}__{digest}"


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in str(value)).strip("-")


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _matrix() -> dict[str, Any]:
    return json.loads((ROOT / "evaluation_matrix.json").read_text(encoding="utf-8"))


def _e1_case_ids() -> dict[str, list[str]]:
    for benchmark in _matrix()["benchmarks"]:
        if benchmark["id"] == "cross-benchmark-generality":
            return {key: [str(item) for item in value] for key, value in benchmark["case_ids"].items()}
    raise RuntimeError("cross-benchmark-generality not found in evaluation_matrix.json")


def _multiagentbench_full_cases() -> list[dict[str, Any]]:
    manifest = json.loads(
        (ROOT / "manifests" / "e1_multiagentbench_full_manifest.json").read_text(encoding="utf-8")
    )
    cases = manifest.get("cases", [])
    if not isinstance(cases, list):
        raise RuntimeError("e1_multiagentbench_full_manifest.json cases must be a list")
    counts = Counter(str(item.get("scenario")) for item in cases)
    expected = {"research": 100, "database": 100, "coding": 100, "minecraft": 100}
    if dict(counts) != expected:
        raise RuntimeError(f"unexpected MultiAgentBench manifest counts: {dict(counts)}")
    return cases


def build_plan(*, include_webarena_deferred: bool = True) -> list[Cell]:
    cases = _e1_case_ids()
    cells: list[Cell] = []
    for task in ("alfworld", "officebench"):
        for case_id in cases[task]:
            for host in HOSTS:
                for memory in MEMORY_METHODS:
                    cells.append(Cell(task, task, case_id, host, memory))
    for item in _multiagentbench_full_cases():
        case_id = str(item["case_id"])
        scenario = str(item["scenario"])
        for host in HOSTS:
            for memory in MEMORY_METHODS:
                cells.append(Cell("multiagentbench", scenario, case_id, host, memory))
    if include_webarena_deferred:
        for case_id in cases.get("webarena", []):
            for host in HOSTS:
                for memory in MEMORY_METHODS:
                    cells.append(Cell("webarena", "webarena", case_id, host, memory))
    return cells


def _run_dir(run_id: str) -> Path:
    return RUNS_ROOT / run_id


def _cell_paths(run_root: Path, cell: Cell) -> dict[str, Path]:
    return {
        "result": run_root / "results" / f"{cell.key}.json",
        "log": run_root / "logs" / f"{cell.key}.adapter.log",
        "state": run_root / "state" / f"{cell.key}.state.db",
        "db": run_root / "db" / f"{cell.key}.team-memory.db",
    }


def _attempt_paths(run_root: Path, cell: Cell) -> dict[str, Path]:
    base = _cell_paths(run_root, cell)
    if not base["log"].exists() and not base["result"].exists():
        return base
    attempt = 1
    while True:
        candidate = {
            key: path.with_name(f"{path.stem}.attempt{attempt}{path.suffix}")
            for key, path in base.items()
        }
        if not any(path.exists() for path in candidate.values()):
            return candidate
        attempt += 1


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _validate_result(path: Path, cell: Cell) -> tuple[dict[str, Any] | None, list[str]]:
    if not path.is_file():
        return None, ["missing_result_json"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, [f"invalid_json:{exc}"]
    context = payload.get("context") or {}
    metrics = payload.get("metrics") or {}
    metadata = payload.get("metadata") or {}
    violations: list[str] = []
    expected = {
        "benchmark": "cross-benchmark-generality",
        "task": cell.task,
        "memory_method": cell.memory_method,
        "actor_model": cell.actor_model,
        "sop_model": cell.effective_sop_model,
        "mas_framework": cell.host,
        "seed": cell.seed,
        "ablation": cell.ablation,
        "case_id": cell.case_id,
    }
    for key, expected_value in expected.items():
        actual = context.get(key)
        if str(actual) != str(expected_value):
            violations.append(f"context.{key}={actual!r} expected {expected_value!r}")
    if _num(metrics.get("primary_score")) is None:
        violations.append("missing_numeric_primary_score")
    if _num(metrics.get("case_count")) != 1.0:
        violations.append(f"case_count={metrics.get('case_count')!r}")
    if not payload.get("cases"):
        violations.append("missing_cases")
    official_log = metadata.get("official_log")
    if official_log and not Path(official_log).is_file():
        violations.append("official_log_missing")
    if cell.memory_method == "gmemory":
        if _num(metrics.get("gmemory_retrieval_count")) is not None and _num(metrics.get("gmemory_retrieval_count")) <= 0:
            violations.append("gmemory_retrieval_count_zero")
        if not metadata.get("memory_adapter_identity"):
            violations.append("missing_memory_adapter_identity")
    if cell.memory_method == "team-memory" and not metadata.get("memory_adapter_identity"):
        violations.append("missing_gems_memory_adapter_identity")
    if not metadata.get("host_adapter_identity"):
        violations.append("missing_host_adapter_identity")
    if not metadata.get("host_topology_hash"):
        violations.append("missing_host_topology_hash")
    if violations:
        return None, violations
    return payload, []


def _classify_failure(log_path: Path, error: str) -> tuple[str, str]:
    text = error
    if log_path.is_file():
        text += "\n" + log_path.read_text(encoding="utf-8", errors="replace")[-60000:]
    lowered = text.lower()
    if "aws_excluded" in lowered or "amazonaws" in lowered or " aws " in f" {lowered} ":
        return "infrastructure_deferred", "aws_dependency_detected"
    if "docker compose" in lowered or "start_docker_containers" in lowered:
        return "dependency_blocked", "docker_compose_start_failed"
    if "official dylan" in lowered or "no verified real" in lowered or "has no real injected runtime" in lowered:
        return "implementation_failed", "real_host_or_memory_adapter_missing"
    if "workspace/solution.py" in lowered:
        return "implementation_failed", "marble_coding_solution_artifact_missing"
    if "none" in lowered and "not iterable" in lowered:
        return "implementation_failed", "officebench_empty_model_action"
    return "implementation_failed", "runtime_or_adapter_failure"


def _result_row(cell: Cell, status: str, paths: dict[str, Path], **extra: Any) -> dict[str, Any]:
    return {
        "benchmark": cell.task,
        "scenario": cell.scenario,
        "case_id": cell.case_id,
        "host": cell.host,
        "memory_method": cell.memory_method,
        "actor_model": cell.actor_model,
        "procedural_memory_model": cell.effective_sop_model,
        "seed": cell.seed,
        "ablation": cell.ablation,
        "status": status,
        "result_path": _relative(paths["result"]),
        "official_log_path": extra.pop("official_log_path", ""),
        "adapter_log_path": _relative(paths["log"]),
        "state_db": _relative(paths["state"]),
        "team_memory_db": _relative(paths["db"]),
        **extra,
    }


def _run_cell(run_root: Path, cell: Cell, *, timeout: int) -> dict[str, Any]:
    paths = _cell_paths(run_root, cell)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    if cell.task == "webarena":
        return _result_row(
            cell,
            "infrastructure_deferred",
            paths,
            failure_category="infrastructure_deferred",
            failure_reason=WEB_ARENA_DEFER_REASON,
        )
    if paths["result"].is_file():
        payload, violations = _validate_result(paths["result"], cell)
        if payload is not None:
            return _accepted_row(cell, paths, payload, skipped_existing=True)
        return _result_row(
            cell,
            "implementation_failed",
            paths,
            failure_category="implementation_failed",
            failure_reason="existing_result_schema_invalid",
            schema_violations=violations,
        )
    if paths["log"].exists():
        paths = _attempt_paths(run_root, cell)
    env = os.environ.copy()
    actor_base_url = (
        env.get("TEAM_MEMORY_EVAL_ACTOR_BASE_URL")
        or env.get("OPENAI_BASE_URL")
        or env.get("OPENAI_API_BASE")
        or ""
    )
    actor_api_key = env.get("TEAM_MEMORY_EVAL_ACTOR_API_KEY") or env.get("OPENAI_API_KEY") or ""
    env.update(
        {
            "PYTHONPATH": f"{ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}",
            "OPENAI_BASE_URL": actor_base_url,
            "OPENAI_API_BASE": actor_base_url,
            "TEAM_MEMORY_EVAL_ACTOR_BASE_URL": actor_base_url,
            "TEAM_MEMORY_EVAL_ACTOR_API_KEY": actor_api_key,
            "TEAM_MEMORY_PROJECT_ROOT": str(ROOT),
            "TEAM_MEMORY_EVAL_BENCHMARK": "cross-benchmark-generality",
            "TEAM_MEMORY_EVAL_TASK": cell.task,
            "TEAM_MEMORY_EVAL_METHOD": cell.memory_method,
            "TEAM_MEMORY_EVAL_ACTOR_MODEL": cell.actor_model,
            "TEAM_MEMORY_EVAL_SOP_MODEL": cell.effective_sop_model,
            "TEAM_MEMORY_EVAL_MAS": cell.host,
            "TEAM_MEMORY_EVAL_ABLATION": cell.ablation,
            "TEAM_MEMORY_EVAL_SEED": str(cell.seed),
            "TEAM_MEMORY_EVAL_CASE_ID": cell.case_id,
            "TEAM_MEMORY_EVAL_OUTPUT": str(paths["result"].resolve()),
            "TEAM_MEMORY_EVAL_STATE_DB": str(paths["state"].resolve()),
            "TEAM_MEMORY_DB": str(paths["db"].resolve()),
            "TEAM_MEMORY_E1_RERUN_ROOT": str(run_root.resolve()),
        }
    )
    command = [
        sys.executable,
        "adapters/team_memory_cross_benchmark_adapter.py",
        "--task",
        cell.task,
        "--mas",
        cell.host,
        "--actor-model",
        cell.actor_model,
        "--sop-model",
        cell.effective_sop_model,
        "--memory-method",
        cell.memory_method,
        "--seed",
        str(cell.seed),
        "--case-id",
        cell.case_id,
        "--ablation",
        cell.ablation,
        "--output",
        str(paths["result"]),
    ]
    started = time.time()
    with paths["log"].open("w", encoding="utf-8") as log:
        log.write(f"started_at={datetime.now(UTC).isoformat()}\n")
        log.write(f"command={json.dumps(command, ensure_ascii=False)}\n")
        log.flush()
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            category, reason = _classify_failure(paths["log"], str(exc))
            return _result_row(
                cell,
                category,
                paths,
                failure_category=category,
                failure_reason=f"adapter_timeout:{reason}",
                elapsed_seconds=time.time() - started,
            )
    if completed.returncode != 0:
        category, reason = _classify_failure(paths["log"], f"returncode={completed.returncode}")
        return _result_row(
            cell,
            category,
            paths,
            failure_category=category,
            failure_reason=reason,
            returncode=completed.returncode,
            elapsed_seconds=time.time() - started,
        )
    payload, violations = _validate_result(paths["result"], cell)
    if payload is None:
        category, reason = _classify_failure(paths["log"], ";".join(violations))
        return _result_row(
            cell,
            category,
            paths,
            failure_category=category,
            failure_reason=reason,
            schema_violations=violations,
            elapsed_seconds=time.time() - started,
        )
    return _accepted_row(cell, paths, payload, skipped_existing=False, elapsed_seconds=time.time() - started)


def _accepted_row(
    cell: Cell,
    paths: dict[str, Path],
    payload: dict[str, Any],
    *,
    skipped_existing: bool,
    elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    metrics = payload.get("metrics") or {}
    metadata = payload.get("metadata") or {}
    official_log = str(metadata.get("official_log", ""))
    return _result_row(
        cell,
        "scored",
        paths,
        failure_category="",
        failure_reason="",
        official_score=float(metrics["primary_score"]),
        score_source=metadata.get("official_score_source", "primary_score"),
        case_count=float(metrics["case_count"]),
        tokens=_num(metrics.get("total_tokens"))
        or _num(metrics.get("actor_prompt_tokens"))
        or _num(metrics.get("marble_token_usage")),
        latency=_num(metrics.get("latency"))
        or _num(metrics.get("latency_seconds"))
        or _num(metrics.get("runtime_seconds")),
        retrieval_count=_num(metrics.get("gmemory_retrieval_count"))
        or _num(metrics.get("sop_retrieval_count"))
        or 0.0,
        adoption_count=_num(metrics.get("sop_adoption_count")) or 0.0,
        recovery_verified_count=_num(metrics.get("recovery_verified_count")) or 0.0,
        snapshot=metadata.get("gmemory_snapshot_id", ""),
        code_revision=_git_revision(ROOT),
        official_log_path=official_log,
        host_adapter_identity=metadata.get("host_adapter_identity", ""),
        host_topology_hash=metadata.get("host_topology_hash", ""),
        memory_adapter_identity=metadata.get("memory_adapter_identity", ""),
        skipped_existing=skipped_existing,
        elapsed_seconds=elapsed_seconds,
    )


def _git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    except Exception:
        return "unknown"


def _write_outputs(run_root: Path, plan: list[Cell], rows: list[dict[str, Any]], run_id: str) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    attempt_rows = list(rows)
    latest_by_key: dict[str, dict[str, Any]] = {}
    for row in attempt_rows:
        latest_by_key[_row_key_from_row(row)] = row
    rows = list(latest_by_key.values())
    case_csv = run_root / "e1_nonaws_case_results.csv"
    summary_json = run_root / "e1_nonaws_table1_summary.json"
    summary_md = run_root / "e1_nonaws_table1_summary.md"
    quality_json = run_root / "e1_nonaws_quality.json"
    checkpoint = run_root / "checkpoint.json"
    table_summary_md = TABLE_DIR / "e1_nonaws_rerun_table1_summary.md"
    table_quality_json = TABLE_DIR / "e1_nonaws_rerun_quality.json"

    fieldnames = [
        "benchmark",
        "scenario",
        "case_id",
        "host",
        "memory_method",
        "actor_model",
        "procedural_memory_model",
        "seed",
        "status",
        "official_score",
        "score_source",
        "tokens",
        "latency",
        "retrieval_count",
        "adoption_count",
        "recovery_verified_count",
        "snapshot",
        "code_revision",
        "result_path",
        "official_log_path",
        "adapter_log_path",
        "failure_category",
        "failure_reason",
    ]
    with case_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    scored = [row for row in rows if row["status"] == "scored"]
    group_scores: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in scored:
        group_scores[(row["benchmark"], row["host"], row["memory_method"])].append(float(row["official_score"]))
    baseline: dict[tuple[str, str], float] = {}
    for (benchmark, host, memory), scores in group_scores.items():
        if memory == "no-memory":
            baseline[(benchmark, host)] = statistics.mean(scores)
    summary_rows = []
    for key, scores in sorted(group_scores.items()):
        benchmark, host, memory = key
        mean = statistics.mean(scores)
        summary_rows.append(
            {
                "benchmark": benchmark,
                "host": host,
                "memory_method": memory,
                "scored_cells": len(scores),
                "mean_official_score": mean,
                "delta_vs_no_memory_mean": mean - baseline.get((benchmark, host), mean),
            }
        )

    expected_status = Counter()
    for cell in plan:
        if cell.task == "webarena":
            expected_status["infrastructure_deferred"] += 1
        else:
            expected_status["planned_nonaws"] += 1
    row_counts = Counter(row["status"] for row in rows)
    missing = sorted(set(cell.key for cell in plan) - set(_row_key_from_row(row) for row in rows))
    schema_violations = [
        row for row in rows if row.get("schema_violations") or row.get("failure_reason") == "existing_result_schema_invalid"
    ]
    matched_issues = _matched_case_issues(rows)
    quality = {
        "run_id": run_id,
        "run_root": _relative(run_root),
        "expected_total_with_webarena_deferred": len(plan),
        "expected_nonaws_final_cells": expected_status["planned_nonaws"],
        "expected_webarena_deferred_cells": expected_status["infrastructure_deferred"],
        "completed_rows": len(rows),
        "scored_cells": row_counts["scored"],
        "infrastructure_deferred": row_counts["infrastructure_deferred"],
        "dependency_blocked": row_counts["dependency_blocked"],
        "implementation_failed": row_counts["implementation_failed"],
        "missing_cells": missing,
        "schema_violations": schema_violations,
        "matched_case_issues": matched_issues,
        "runtime_failures": [
            row for row in rows if row["status"] in {"implementation_failed", "dependency_blocked"}
        ],
        "attempt_rows": attempt_rows,
        "summary_rows": summary_rows,
        "case_csv": _relative(case_csv),
        "summary_md": _relative(summary_md),
        "summary_json": _relative(summary_json),
    }
    summary_json.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    quality_json.write_text(json.dumps(quality, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    checkpoint.write_text(json.dumps({"run_id": run_id, "rows": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# E1 Non-AWS Table 1 Rerun Summary",
        "",
        f"- Run ID: `{run_id}`",
        f"- Run root: `{_relative(run_root)}`",
        f"- Non-AWS final cells scored: {row_counts['scored']} / {expected_status['planned_nonaws']}",
        f"- WebArena deferred cells: {row_counts['infrastructure_deferred']} / {expected_status['infrastructure_deferred']}",
        f"- Dependency blocked: {row_counts['dependency_blocked']}",
        f"- Implementation failed: {row_counts['implementation_failed']}",
        "",
        "## Mean Scores",
        "",
        "| Benchmark | Host | Memory | Scored cells | Mean official score | Delta vs no-memory |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {benchmark} | {host} | {memory_method} | {scored_cells} | {mean:.3f} | {delta:.3f} |".format(
                mean=row["mean_official_score"],
                delta=row["delta_vs_no_memory_mean"],
                **row,
            )
        )
    lines.extend(
        [
            "",
            "## Blockers",
            "",
            "| Benchmark | Scenario | Host | Memory | Case | Status | Reason | Repair command |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        if row["status"] == "scored":
            continue
        lines.append(
            "| {benchmark} | {scenario} | {host} | {memory_method} | {case_id} | {status} | {reason} | `{cmd}` |".format(
                reason=row.get("failure_reason", ""),
                cmd=_repair_command(row, run_id),
                **row,
            )
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    table_summary_md.write_text(summary_md.read_text(encoding="utf-8"), encoding="utf-8")
    table_quality_json.write_text(quality_json.read_text(encoding="utf-8"), encoding="utf-8")


def _row_key_from_row(row: dict[str, Any]) -> str:
    return Cell(
        benchmark=row["benchmark"],
        scenario=row["scenario"],
        case_id=row["case_id"],
        host=row["host"],
        memory_method=row["memory_method"],
        actor_model=row.get("actor_model", "gpt-5-mini"),
        sop_model="gpt-5-mini",
        seed=int(row.get("seed", 0)),
        ablation=row.get("ablation", "full"),
    ).key


def _matched_case_issues(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        if row["status"] != "scored":
            continue
        grouped[(row["benchmark"], row["host"])][row["memory_method"]].add(row["case_id"])
    for group, methods in grouped.items():
        if set(methods) != set(MEMORY_METHODS):
            issues.append({"group": list(group), "reason": "not_all_methods_scored", "methods": sorted(methods)})
            continue
        case_sets = list(methods.values())
        if any(case_set != case_sets[0] for case_set in case_sets[1:]):
            issues.append(
                {
                    "group": list(group),
                    "reason": "scored_case_sets_differ",
                    "case_sets": {method: sorted(cases) for method, cases in methods.items()},
                }
            )
    return issues


def _repair_command(row: dict[str, Any], run_id: str) -> str:
    if row["status"] == "infrastructure_deferred":
        return "AWS excluded by user; re-enable WebArena only after explicit user instruction."
    return (
        f"python scripts/run_e1_nonaws_rerun.py --run-id {run_id} "
        f"--task {row['benchmark']} --host {row['host']} "
        f"--memory-method {row['memory_method']} --case-id {row['case_id']}"
    )


def _load_existing_rows(run_root: Path) -> list[dict[str, Any]]:
    checkpoint = run_root / "checkpoint.json"
    if not checkpoint.is_file():
        return []
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    return rows if isinstance(rows, list) else []


def _select_plan(plan: list[Cell], args: argparse.Namespace) -> list[Cell]:
    selected = []
    for cell in plan:
        if args.stage == "alfworld-officebench" and cell.task not in {"alfworld", "officebench", "webarena"}:
            continue
        if args.stage == "alfworld" and cell.task not in {"alfworld", "webarena"}:
            continue
        if args.stage == "officebench" and cell.task not in {"officebench", "webarena"}:
            continue
        if args.stage == "multiagentbench" and cell.task not in {"multiagentbench", "webarena"}:
            continue
        if args.task and cell.task != args.task:
            continue
        if args.host and cell.host != args.host:
            continue
        if args.memory_method and cell.memory_method != args.memory_method:
            continue
        if args.case_id and cell.case_id != args.case_id:
            continue
        if cell.task == "webarena" and args.no_webarena_deferred_rows:
            continue
        selected.append(cell)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=os.environ.get("TEAM_MEMORY_E1_NONAWS_RUN_ID"))
    parser.add_argument("--stage", choices=("all", "alfworld", "officebench", "alfworld-officebench", "multiagentbench"), default="all")
    parser.add_argument("--task")
    parser.add_argument("--host")
    parser.add_argument("--memory-method")
    parser.add_argument("--case-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=int(os.environ.get("TEAM_MEMORY_E1_FINAL_TIMEOUT_SECONDS", "7200")))
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--no-webarena-deferred-rows", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    _load_env_file(ROOT / args.env_file)
    run_id = args.run_id or datetime.now(UTC).strftime("e1-nonaws-%Y%m%dT%H%M%SZ")
    run_root = _run_dir(run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "frozen_config.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_or_resumed_at": datetime.now(UTC).isoformat(),
                "actor_model": "gpt-5-mini",
                "procedural_memory_model": "gpt-5-mini",
                "hosts": HOSTS,
                "memory_methods": MEMORY_METHODS,
                "seed": 0,
                "web_arena_policy": WEB_ARENA_DEFER_REASON,
                "code_revision": _git_revision(ROOT),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    plan = _select_plan(build_plan(), args)
    if args.limit is not None:
        plan = plan[: args.limit]
    (run_root / "manifest.json").write_text(
        json.dumps([cell.__dict__ | {"key": cell.key} for cell in plan], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rows = _load_existing_rows(run_root)
    seen = {
        _row_key_from_row(row)
        for row in rows
        if row.get("status") == "scored"
        or (row.get("benchmark") == "webarena" and row.get("status") == "infrastructure_deferred")
    }
    if args.report_only:
        _write_outputs(run_root, plan, rows, run_id)
        print(f"REPORT {run_root}")
        return 0
    for index, cell in enumerate(plan, start=1):
        if cell.key in seen:
            continue
        print(f"RUN {index}/{len(plan)} {cell.task} {cell.scenario} {cell.host} {cell.memory_method} {cell.case_id}", flush=True)
        row = _run_cell(run_root, cell, timeout=args.timeout_seconds)
        rows.append(row)
        seen.add(cell.key)
        print(
            f"{row['status'].upper()} {cell.task} {cell.host} {cell.memory_method} {cell.case_id} "
            f"{row.get('official_score', row.get('failure_reason', ''))}",
            flush=True,
        )
        _write_outputs(run_root, plan, rows, run_id)
    _write_outputs(run_root, plan, rows, run_id)
    print(f"WROTE {_relative(run_root / 'e1_nonaws_table1_summary.md')}", flush=True)
    print(f"WROTE {_relative(run_root / 'e1_nonaws_quality.json')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
