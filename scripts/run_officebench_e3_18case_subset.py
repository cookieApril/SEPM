#!/usr/bin/env python3
"""Run the configured OfficeBench E3 18-case subset with stop-on-failure checks."""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "benchmark-results" / "unified-v3" / "results"
TABLE_DIR = ROOT / "benchmark-results" / "unified-v3" / "tables"
LOG_DIR = ROOT / "benchmark-results" / "unified-v3" / "logs"
SMOKE_DB_DIR = ROOT / "benchmark-results" / "smoke-db"
SMOKE_STATE_DIR = ROOT / "benchmark-results" / "smoke-state"

ABLATIONS = ["no-extra-components", "sop-only", "divergence-only", "full"]
SUMMARY_MD = TABLE_DIR / "officebench_e3_18case_summary.md"
QUALITY_JSON = TABLE_DIR / "officebench_e3_18case_quality.json"
RUN_LOG = LOG_DIR / "officebench_e3_18case_runner.log"


def _load_officebench_case_ids() -> list[str]:
    override = os.environ.get("OFFICEBENCH_E3_CASE_IDS", "").strip()
    if override:
        return [item.strip() for item in override.split(",") if item.strip()]
    matrix = json.loads((ROOT / "evaluation_matrix.json").read_text())
    benchmark = next(item for item in matrix["benchmarks"] if item["id"] == "component-ablation")
    case_ids = benchmark["case_ids"]["officebench"]
    if len(case_ids) != 18:
        raise RuntimeError(f"Expected 18 OfficeBench E3 cases, found {len(case_ids)}")
    return case_ids


def _load_officebench_manifest_metadata() -> dict[str, object]:
    manifest_path = ROOT / "manifests" / "e3_case_manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text())
    officebench = manifest.get("benchmarks", {}).get("officebench", {})
    replacements = [
        {
            "case_id": item.get("case_id"),
            "replacement_for": item.get("replacement_for"),
            "selection_hash": item.get("selection_hash"),
            "selection_hash_salt": item.get("selection_hash_salt"),
            "note": item.get("note"),
        }
        for item in officebench.get("cases", [])
        if item.get("replacement_for")
    ]
    return {
        "excluded_official_evaluator_bug_cases": officebench.get("excluded_cases", []),
        "replacement_case_records": replacements,
        "manifest_notes": officebench.get("notes", []),
    }


def _safe_case(case_id: str) -> str:
    return case_id.replace("/", "-").replace(" ", "_")


def _result_glob(case_id: str, ablation: str) -> list[Path]:
    safe = _safe_case(case_id)
    pattern = (
        "component-ablation__officebench__team-memory__gpt-5-mini__gpt-5-mini"
        f"__autogen__{ablation}__s0__{safe}__*.json"
    )
    return [
        path
        for path in RESULT_DIR.glob(pattern)
        if not path.name.endswith(".team-memory-runtime-metrics.json")
    ]


def _latest_result(case_id: str, ablation: str, started_at: float) -> Path:
    candidates = _result_glob(case_id, ablation)
    fresh = [path for path in candidates if path.stat().st_mtime >= started_at - 1.0]
    if not fresh:
        raise RuntimeError(f"No fresh official result JSON for {case_id} {ablation}")
    return max(fresh, key=lambda path: path.stat().st_mtime)


def _latest_existing_result(case_id: str, ablation: str) -> Path | None:
    candidates = _result_glob(case_id, ablation)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _num(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _check_result(path: Path, case_id: str, ablation: str) -> dict[str, object]:
    data = json.loads(path.read_text())
    metrics = data.get("metrics") or {}
    schema: list[str] = []
    if "context" not in data:
        schema.append("missing context")
    if "primary_score" not in metrics:
        schema.append("missing metrics.primary_score")
    if _num(metrics.get("case_count"), -1.0) != 1.0:
        schema.append(f"case_count={metrics.get('case_count')}")
    if not isinstance(data.get("cases"), list) or not data.get("cases"):
        schema.append("missing/nonempty cases")

    switch: list[str] = []
    team_memory = _num(metrics.get("team_memory_enabled"))
    procedural = _num(metrics.get("procedural_memory_enabled"))
    divergence = _num(metrics.get("divergence_enabled"))
    blackboard = _num(metrics.get("blackboard_entries"))
    if ablation == "no-extra-components":
        if team_memory != 0.0:
            switch.append(f"team_memory_enabled={team_memory}")
        if procedural != 0.0:
            switch.append(f"procedural_memory_enabled={procedural}")
        if divergence != 0.0:
            switch.append(f"divergence_enabled={divergence}")
        if blackboard != 0.0:
            switch.append(f"blackboard_entries={blackboard}")
    elif ablation == "sop-only":
        if procedural != 1.0:
            switch.append(f"procedural_memory_enabled={procedural}")
        if divergence != 0.0:
            switch.append(f"divergence_enabled={divergence}")
    elif ablation == "divergence-only":
        if procedural != 0.0:
            switch.append(f"procedural_memory_enabled={procedural}")
        if divergence != 1.0:
            switch.append(f"divergence_enabled={divergence}")
    elif ablation == "full":
        if procedural != 1.0:
            switch.append(f"procedural_memory_enabled={procedural}")
        if divergence != 1.0:
            switch.append(f"divergence_enabled={divergence}")

    if schema or switch:
        raise RuntimeError(
            f"Result validation failed for {case_id} {ablation}: "
            f"schema={schema or 'ok'} switch={switch or 'ok'}"
        )

    latency = metrics.get("latency", metrics.get("latency_seconds", metrics.get("duration_seconds", 0.0)))
    return {
        "benchmark": "OfficeBench",
        "case_id": case_id,
        "case_id_safe": _safe_case(case_id),
        "condition": ablation,
        "official_score": _num(metrics.get("primary_score")),
        "case_count": _num(metrics.get("case_count")),
        "team_memory_enabled": team_memory,
        "procedural_memory_enabled": procedural,
        "divergence_enabled": divergence,
        "blackboard_entries": _num(metrics.get("blackboard_entries")),
        "sop_retrieval_count": _num(metrics.get("sop_retrieval_count")),
        "divergence_events": _num(metrics.get("divergence_events")),
        "unsafe_accepted": _num(metrics.get("unsafe_accepted")),
        "prompt_char_count_max": _num(metrics.get("prompt_char_count_max")),
        "observation_truncated_count": _num(metrics.get("observation_truncated_count")),
        "largest_observation_chars": _num(metrics.get("largest_observation_chars")),
        "tool_output_truncated_count": _num(metrics.get("tool_output_truncated_count")),
        "latency": _num(latency),
        "result_path": str(path.relative_to(ROOT)),
    }


def _write_outputs(rows: list[dict[str, object]], stopped: dict[str, object] | None = None) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    by_case_no_extra = {
        str(row["case_id"]): _num(row["official_score"])
        for row in rows
        if row["condition"] == "no-extra-components"
    }
    for row in rows:
        base = by_case_no_extra.get(str(row["case_id"]))
        row["delta_vs_no_extra"] = None if base is None else _num(row["official_score"]) - base

    headers = [
        "Benchmark",
        "Case ID",
        "Condition",
        "Official score",
        "Delta vs no-extra",
        "Prompt max chars",
        "Obs truncated",
        "Largest obs chars",
        "Tool output truncated",
        "Blackboard entries",
        "SOP retrieval count",
        "Divergence events",
        "Unsafe accepted",
        "Latency",
        "Result path",
    ]
    keys = [
        "benchmark",
        "case_id",
        "condition",
        "official_score",
        "delta_vs_no_extra",
        "prompt_char_count_max",
        "observation_truncated_count",
        "largest_observation_chars",
        "tool_output_truncated_count",
        "blackboard_entries",
        "sop_retrieval_count",
        "divergence_events",
        "unsafe_accepted",
        "latency",
        "result_path",
    ]
    lines = [
        "# OfficeBench E3 18-Case Matched Subset Summary",
        "",
        "This table contains the configured OfficeBench E3 18-case matched subset only; it is not an official full-benchmark sweep.",
        "",
    ]
    if stopped:
        lines.extend(["## Stop Status", "", f"- Stopped: `{stopped}`", ""])
    lines.extend(["## Results", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"])
    for row in rows:
        values = []
        for key in keys:
            value = row.get(key)
            if value is None:
                value = ""
            values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")

    success_by_condition = {
        ablation: sum(1 for row in rows if row["condition"] == ablation and _num(row["official_score"]) > 0)
        for ablation in ABLATIONS
    }
    full_compare = {"improved": 0, "regressed": 0, "tie": 0}
    full_by_case = {str(row["case_id"]): _num(row["official_score"]) for row in rows if row["condition"] == "full"}
    for case_id, base in by_case_no_extra.items():
        if case_id not in full_by_case:
            continue
        diff = full_by_case[case_id] - base
        if diff > 0:
            full_compare["improved"] += 1
        elif diff < 0:
            full_compare["regressed"] += 1
        else:
            full_compare["tie"] += 1

    def metric_summary(name: str) -> dict[str, float]:
        values = [_num(row.get(name)) for row in rows]
        if not values:
            return {"mean": 0.0, "max": 0.0}
        return {"mean": statistics.mean(values), "max": max(values)}

    quality = {
        "completed_rows": len(rows),
        "expected_rows": 18 * 4,
        "stopped": stopped,
        "schema_violations": 0,
        "switch_violations": 0,
        "success_by_condition": success_by_condition,
        "full_vs_no_extra": full_compare,
        "truncation_metrics": {
            "prompt_char_count_max": metric_summary("prompt_char_count_max"),
            "observation_truncated_count": metric_summary("observation_truncated_count"),
            "largest_observation_chars": metric_summary("largest_observation_chars"),
            "tool_output_truncated_count": metric_summary("tool_output_truncated_count"),
        },
        "rows": rows,
    }
    quality.update(_load_officebench_manifest_metadata())

    lines.extend(
        [
            "",
            "## Quality",
            "",
            f"- Completed official result JSON rows: {len(rows)}/72",
            "- Schema violations: 0",
            "- Switch violations: 0",
            f"- Success by condition: {success_by_condition}",
            f"- Full vs no-extra: {full_compare}",
        ]
    )
    for metric, stats in quality["truncation_metrics"].items():
        lines.append(f"- {metric}: mean={stats['mean']:.2f}, max={stats['max']:.0f}")

    SUMMARY_MD.write_text("\n".join(lines) + "\n")
    QUALITY_JSON.write_text(json.dumps(quality, indent=2, ensure_ascii=False) + "\n")


def _append_run_log(text: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def main() -> int:
    os.chdir(ROOT)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SMOKE_DB_DIR.mkdir(parents=True, exist_ok=True)
    SMOKE_STATE_DIR.mkdir(parents=True, exist_ok=True)
    RUN_LOG.write_text("")

    case_ids = _load_officebench_case_ids()
    rows: list[dict[str, object]] = []
    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = "src"
    skip_existing = os.environ.get("OFFICEBENCH_E3_SKIP_EXISTING", "").strip() == "1"

    for case_id in case_ids:
        safe = _safe_case(case_id)
        for ablation in ABLATIONS:
            if skip_existing:
                existing = _latest_existing_result(case_id, ablation)
                if existing is not None:
                    try:
                        row = _check_result(existing, case_id, ablation)
                    except Exception as exc:
                        stopped = {
                            "case_id": case_id,
                            "ablation": ablation,
                            "reason": "existing_result_validation_failed",
                            "error": str(exc),
                        }
                        _write_outputs(rows, stopped=stopped)
                        print(json.dumps(stopped, indent=2), flush=True)
                        return 3
                    rows.append(row)
                    print(
                        f"SKIP {case_id} {ablation} existing score={row['official_score']} "
                        f"prompt_max={row['prompt_char_count_max']} obs_trunc={row['observation_truncated_count']}",
                        flush=True,
                    )
                    continue
            team_db = SMOKE_DB_DIR / f"officebench-e3-18case-s0-{safe}-{ablation}.team-memory.db"
            state_db = SMOKE_STATE_DIR / f"officebench-e3-18case-s0-{safe}-{ablation}.state.db"
            env = env_base.copy()
            env["TEAM_MEMORY_DB"] = str(team_db)
            cmd = [
                sys.executable,
                "-m",
                "team_memory.evaluation_runner",
                "run",
                "--benchmark",
                "component-ablation",
                "--task",
                "officebench",
                "--memory-method",
                "team-memory",
                "--actor-model",
                "gpt-5-mini",
                "--sop-model",
                "gpt-5-mini",
                "--mas",
                "autogen",
                "--seed",
                "0",
                "--case-id",
                case_id,
                "--ablation",
                ablation,
                "--state-db",
                str(state_db),
                "--no-resume",
                "--rerun-failed",
            ]
            print(f"RUN {case_id} {ablation}", flush=True)
            started_at = time.time()
            proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True)
            _append_run_log(f"\n===== {case_id} {ablation} rc={proc.returncode} =====")
            _append_run_log(proc.stdout[-20000:])
            _append_run_log(proc.stderr[-20000:])
            if proc.returncode != 0:
                stopped = {
                    "case_id": case_id,
                    "ablation": ablation,
                    "reason": "runner_nonzero",
                    "returncode": proc.returncode,
                    "stdout_tail": proc.stdout[-2000:],
                    "stderr_tail": proc.stderr[-2000:],
                }
                _write_outputs(rows, stopped=stopped)
                print(json.dumps(stopped, indent=2), flush=True)
                return 2
            try:
                result_path = _latest_result(case_id, ablation, started_at)
                row = _check_result(result_path, case_id, ablation)
            except Exception as exc:  # stop-on-failure gate
                stopped = {
                    "case_id": case_id,
                    "ablation": ablation,
                    "reason": "result_validation_failed",
                    "error": str(exc),
                }
                _write_outputs(rows, stopped=stopped)
                print(json.dumps(stopped, indent=2), flush=True)
                return 3
            rows.append(row)
            print(
                f"OK {case_id} {ablation} score={row['official_score']} "
                f"prompt_max={row['prompt_char_count_max']} obs_trunc={row['observation_truncated_count']}",
                flush=True,
            )

    _write_outputs(rows)
    print(f"WROTE {SUMMARY_MD.relative_to(ROOT)}", flush=True)
    print(f"WROTE {QUALITY_JSON.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
