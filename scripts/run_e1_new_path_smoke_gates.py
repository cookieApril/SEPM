#!/usr/bin/env python
"""Run bounded official smoke gates for newly wired E1 host/memory paths.

This script does not expand the final E1 matrix.  It runs only explicit
single-case development/smoke cells and refuses to overwrite accepted paper
results because outputs live under ``benchmark-results/smoke/e1-new-paths``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_ROOT = PROJECT_ROOT / "benchmark-results/smoke/e1-new-paths"
REPORT_DIR = PROJECT_ROOT / "benchmark-results/unified-v3/tables"
REPORT_JSON = REPORT_DIR / "e1_new_path_smoke_gate_report.json"
REPORT_MD = REPORT_DIR / "e1_new_path_smoke_gate_report.md"


SMOKE_CELLS = [
    ("webarena", "autogen", "gmemory", "219"),
    ("officebench", "autogen", "gmemory", "2-39/1"),
    ("multiagentbench", "autogen", "no-memory", "research/47"),
    ("multiagentbench", "autogen", "gmemory", "research/47"),
    ("multiagentbench", "autogen", "team-memory", "research/47"),
    ("webarena", "dylan", "no-memory", "219"),
    ("webarena", "dylan", "gmemory", "219"),
    ("webarena", "dylan", "team-memory", "219"),
    ("officebench", "dylan", "no-memory", "2-39/1"),
    ("officebench", "dylan", "gmemory", "2-39/1"),
    ("officebench", "dylan", "team-memory", "2-39/1"),
    ("multiagentbench", "dylan", "no-memory", "research/47"),
    ("multiagentbench", "dylan", "gmemory", "research/47"),
    ("multiagentbench", "dylan", "team-memory", "research/47"),
    ("multiagentbench", "autogen", "no-memory", "database/65"),
    ("multiagentbench", "autogen", "gmemory", "database/65"),
    ("multiagentbench", "autogen", "team-memory", "database/65"),
    ("multiagentbench", "dylan", "no-memory", "database/65"),
    ("multiagentbench", "dylan", "gmemory", "database/65"),
    ("multiagentbench", "dylan", "team-memory", "database/65"),
    ("multiagentbench", "autogen", "no-memory", "coding/63"),
    ("multiagentbench", "autogen", "gmemory", "coding/63"),
    ("multiagentbench", "autogen", "team-memory", "coding/63"),
    ("multiagentbench", "dylan", "no-memory", "coding/63"),
    ("multiagentbench", "dylan", "gmemory", "coding/63"),
    ("multiagentbench", "dylan", "team-memory", "coding/63"),
    ("multiagentbench", "autogen", "no-memory", "minecraft/9"),
    ("multiagentbench", "autogen", "gmemory", "minecraft/9"),
    ("multiagentbench", "autogen", "team-memory", "minecraft/9"),
    ("multiagentbench", "dylan", "no-memory", "minecraft/9"),
    ("multiagentbench", "dylan", "gmemory", "minecraft/9"),
    ("multiagentbench", "dylan", "team-memory", "minecraft/9"),
]


def _slug(*parts: str) -> str:
    text = "__".join(part.replace("/", "-") for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{text}__{digest}"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cell_status(result_path: Path, log_path: Path) -> dict[str, Any]:
    official_logs = sorted(result_path.parent.glob(f"{result_path.stem}.*/*official-runtime.log"))
    official_logs.extend(sorted(result_path.parent.glob(f"{result_path.stem}.officebench.log")))
    official_log = official_logs[0] if official_logs else log_path
    if not result_path.is_file():
        error_files = sorted(result_path.parent.glob(f"{result_path.stem}.*/error.txt"))
        category = "implementation_failed" if official_log != log_path else "not_run"
        reason = "No smoke result JSON was produced."
        if official_log.is_file():
            text = official_log.read_text(encoding="utf-8", errors="replace")[-40000:].lower()
            if "docker compose" in text or "start_docker_containers" in text:
                category = "dependency_blocked"
                reason = "MARBLE Database Docker compose startup failed before official scoring."
            elif "failed to read code from workspace/solution.py" in text:
                reason = "MARBLE Coding did not produce the required workspace/solution.py artifact."
            elif "nonetype" in text and "not iterable" in text:
                reason = "OfficeBench policy received an empty/None model action before official scoring."
            elif "connection refused" in text and "localhost" in text:
                reason = "Local benchmark service connection was refused before accepted scoring."
        return {
            "status": category if category != "not_run" else "missing",
            "failure_category": category,
            "failure_reason": reason,
            "result_path": str(result_path),
            "adapter_log": str(log_path),
            "official_log": str(official_log),
            "error_path": str(error_files[0]) if error_files else "",
            "gate_passed": False,
        }
    payload = _load_json(result_path)
    metrics = payload.get("metrics", {})
    metadata = payload.get("metadata", {})
    passed = (
        float(metrics.get("case_count", 0.0)) == 1.0
        and "primary_score" in metrics
        and official_log.is_file()
        and official_log.stat().st_size > 0
        and metadata.get("host_adapter_identity")
        and metadata.get("host_topology_hash")
        and metadata.get("memory_adapter_identity") is not None
    )
    if payload.get("context", {}).get("memory_method") in {"gmemory", "team-memory"}:
        if payload["context"]["memory_method"] == "gmemory":
            passed = passed and float(metrics.get("gmemory_retrieval_count", 0.0)) > 0
        else:
            passed = passed and float(metrics.get("team_memory_enabled", 0.0)) > 0
    host_trace = metadata.get("host_trace", "")
    if host_trace:
        passed = passed and Path(host_trace).is_file() and Path(host_trace).stat().st_size > 0
    return {
        "status": "accepted" if passed else "implementation_failed",
        "failure_category": "accepted" if passed else "implementation_failed",
        "gate_passed": bool(passed),
        "result_path": str(result_path),
        "adapter_log": str(log_path),
        "official_log": str(official_log),
        "score": metrics.get("primary_score"),
        "score_source": metadata.get("official_score_source", "primary_score"),
        "host_identity": metadata.get("host_adapter_identity"),
        "memory_identity": metadata.get("memory_adapter_identity"),
        "host_trace_path": host_trace,
        "retrieval_evidence_path": _first_glob(
            result_path.parent, f"{result_path.stem}.*/gmemory-retrieval-evidence.json"
        ),
        "retrieval_count": metrics.get("gmemory_retrieval_count", metrics.get("sop_retrieval_count", 0.0)),
    }


def _latest_attempt(slug: str) -> tuple[Path, Path]:
    base_result_path = SMOKE_ROOT / f"{slug}.json"
    base_log_path = SMOKE_ROOT / f"{slug}.adapter.log"
    if base_result_path.exists():
        return base_result_path, base_log_path
    attempts: list[tuple[int, Path, Path]] = []
    for path in SMOKE_ROOT.glob(f"{slug}.attempt*.json"):
        attempt_text = path.stem.rsplit(".attempt", 1)[-1]
        if attempt_text.isdigit():
            attempts.append((int(attempt_text), path, SMOKE_ROOT / f"{slug}.attempt{attempt_text}.adapter.log"))
    if attempts:
        _, result_path, log_path = sorted(attempts)[-1]
        return result_path, log_path
    log_attempts: list[tuple[int, Path, Path]] = []
    for path in SMOKE_ROOT.glob(f"{slug}.attempt*.adapter.log"):
        attempt_text = path.stem.rsplit(".attempt", 1)[-1].split(".", 1)[0]
        if attempt_text.isdigit():
            result_path = SMOKE_ROOT / f"{slug}.attempt{attempt_text}.json"
            log_attempts.append((int(attempt_text), result_path, path))
    if log_attempts:
        _, result_path, log_path = sorted(log_attempts)[-1]
        return result_path, log_path
    return base_result_path, base_log_path


def _next_attempt_paths(slug: str) -> tuple[Path, Path]:
    base_result_path = SMOKE_ROOT / f"{slug}.json"
    base_log_path = SMOKE_ROOT / f"{slug}.adapter.log"
    if not base_log_path.exists() and not any(SMOKE_ROOT.glob(f"{slug}.attempt*.adapter.log")):
        return base_result_path, base_log_path
    attempt = 1
    while (
        (SMOKE_ROOT / f"{slug}.attempt{attempt}.json").exists()
        or (SMOKE_ROOT / f"{slug}.attempt{attempt}.adapter.log").exists()
    ):
        attempt += 1
    return SMOKE_ROOT / f"{slug}.attempt{attempt}.json", SMOKE_ROOT / f"{slug}.attempt{attempt}.adapter.log"


def _write_report(rows: list[dict[str, Any]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "expected": len(rows),
        "completed": sum(1 for row in rows if row.get("gate_passed")),
        "full_e1_launched": False,
        "rows": rows,
    }
    REPORT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# E1 New Path Smoke Gate Report",
        "",
        f"- Expected smoke cells: {payload['expected']}",
        f"- Passed smoke cells: {payload['completed']}",
        "- Final 2,592-cell sweep launched: no",
        "",
        "| Task | MAS | Memory | Case | Status | Category | Score | Result | Log | Reason | Repair / rerun |",
        "|---|---|---|---|---|---|---:|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {task} | {mas} | {memory_method} | {case_id} | {status} | {failure_category} | {score} | {result_path} | {official_log} | {failure_reason} | {repair_command} |".format(
                **{**row, "score": row.get("score", "n/a"), "failure_reason": row.get("failure_reason", "")}
            )
        )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_cell(task: str, mas: str, memory: str, case_id: str) -> dict[str, Any]:
    slug = _slug(task, mas, memory, case_id)
    latest_result_path, latest_log_path = _latest_attempt(slug)
    if latest_result_path.exists():
        return {
            "task": task,
            "mas": mas,
            "memory_method": memory,
            "case_id": case_id,
            "skipped_existing": True,
            **_cell_status(latest_result_path, latest_log_path),
            "repair_command": "",
        }
    result_path, log_path = _next_attempt_paths(slug)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": f"{PROJECT_ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}",
            "TEAM_MEMORY_PROJECT_ROOT": str(PROJECT_ROOT),
            "TEAM_MEMORY_EVAL_BENCHMARK": "cross-benchmark-generality",
            "TEAM_MEMORY_EVAL_TASK": task,
            "TEAM_MEMORY_EVAL_METHOD": memory,
            "TEAM_MEMORY_EVAL_ACTOR_MODEL": "gpt-5-mini",
            "TEAM_MEMORY_EVAL_SOP_MODEL": "not-applicable" if memory != "team-memory" else "gpt-5-mini",
            "TEAM_MEMORY_EVAL_MAS": mas,
            "TEAM_MEMORY_EVAL_ABLATION": "full",
            "TEAM_MEMORY_EVAL_SEED": "0",
            "TEAM_MEMORY_EVAL_CASE_ID": case_id,
            "TEAM_MEMORY_EVAL_OUTPUT": str(result_path.resolve()),
            "TEAM_MEMORY_DB": str((SMOKE_ROOT / f"{slug}.team-memory.db").resolve()),
            "TEAM_MEMORY_EVAL_STATE_DB": str((SMOKE_ROOT / f"{slug}.state.db").resolve()),
        }
    )
    command = [
        sys.executable,
        "adapters/team_memory_cross_benchmark_adapter.py",
        "--task",
        task,
        "--mas",
        mas,
        "--actor-model",
        "gpt-5-mini",
        "--sop-model",
        env["TEAM_MEMORY_EVAL_SOP_MODEL"],
        "--memory-method",
        memory,
        "--seed",
        "0",
        "--case-id",
        case_id,
        "--ablation",
        "full",
        "--output",
        str(result_path),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=int(os.environ.get("TEAM_MEMORY_E1_SMOKE_TIMEOUT_SECONDS", "7200")),
            check=False,
        )
    row = {
        "task": task,
        "mas": mas,
        "memory_method": memory,
        "case_id": case_id,
        "returncode": completed.returncode,
        "skipped_existing": False,
        **_cell_status(result_path, log_path),
        "repair_command": _rerun_command(task, mas, memory, case_id),
    }
    if completed.returncode != 0 or not row["gate_passed"]:
        row["runtime_failure"] = True
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task")
    parser.add_argument("--mas")
    parser.add_argument("--memory-method")
    parser.add_argument("--case-id")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--skip-task", action="append", default=[])
    parser.add_argument("--continue-on-failure", action="store_true")
    args = parser.parse_args()
    cells = [
        cell
        for cell in SMOKE_CELLS
        if (not args.task or cell[0] == args.task)
        and cell[0] not in set(args.skip_task)
        and (not args.mas or cell[1] == args.mas)
        and (not args.memory_method or cell[2] == args.memory_method)
        and (not args.case_id or cell[3] == args.case_id)
    ]
    if args.limit is not None:
        cells = cells[: args.limit]
    rows: list[dict[str, Any]] = []
    for task, mas, memory, case_id in SMOKE_CELLS:
        if task in set(args.skip_task):
            slug = _slug(task, mas, memory, case_id)
            result_path, log_path = _latest_attempt(slug)
            row = {
                "task": task,
                "mas": mas,
                "memory_method": memory,
                "case_id": case_id,
                "skipped_existing": False,
                "status": "infrastructure_deferred",
                "failure_category": "infrastructure_deferred",
                "result_path": str(result_path),
                "adapter_log": str(log_path),
                "official_log": "",
                "error_path": "",
                "gate_passed": False,
                "repair_command": "Re-enable the benchmark infrastructure, then rerun this smoke slice.",
            }
            rows.append(row)
    for cell in cells:
        if args.report_only:
            task, mas, memory, case_id = cell
            slug = _slug(task, mas, memory, case_id)
            result_path, log_path = _latest_attempt(slug)
            row = {
                "task": task,
                "mas": mas,
                "memory_method": memory,
                "case_id": case_id,
                "skipped_existing": result_path.exists(),
                **_cell_status(result_path, log_path),
                "repair_command": _rerun_command(task, mas, memory, case_id),
            }
        else:
            row = run_cell(*cell)
        rows.append(row)
        _write_report(rows)
        if row.get("runtime_failure") and not args.continue_on_failure:
            return 1
    return 0


def _rerun_command(task: str, mas: str, memory: str, case_id: str) -> str:
    return (
        "python scripts/run_e1_new_path_smoke_gates.py "
        f"--task {task} --mas {mas} --memory-method {memory} --case-id {case_id} --continue-on-failure"
    )


def _first_glob(root: Path, pattern: str) -> str:
    matches = sorted(root.glob(pattern))
    return str(matches[0]) if matches else ""


if __name__ == "__main__":
    raise SystemExit(main())
