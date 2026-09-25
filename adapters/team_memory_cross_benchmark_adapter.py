"""Single-case adapter for the cross-benchmark Team Memory experiments.

The adapter is intentionally conservative: it refuses to run without
``--case-id`` and resolves that id against the downloaded official manifests
before doing any work.  ALFWorld delegates to the existing G-Memory adapter so
the current official ALFWorld/GMemory path stays intact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SUPPORTED_TASKS = ("alfworld", "webarena", "multiagentbench", "officebench")
SUPPORTED_MEMORY_METHODS = (
    "no-memory",
    "agent-native-memory",
    "generative-memory",
    "gmemory",
    "mem0",
    "team-memory",
)
GMEMORY_METHODS = {"no-memory", "gmemory", "team-memory"}
OFFICIAL_RUNTIME_METHODS = {"no-memory", "team-memory"}
NON_ALFWORLD_OFFICIAL_HOSTS = {"autogen"}


def _stable_identity_hash(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _runtime_identity_metadata(
    *,
    args: argparse.Namespace,
    task: str,
    official_entrypoint: str,
) -> dict[str, Any]:
    """Return auditable runtime identity fields for E1 smoke gates."""
    if task != "alfworld" and args.mas not in NON_ALFWORLD_OFFICIAL_HOSTS:
        raise RuntimeError(
            f"{task} has no verified real {args.mas!r} official-loop host adapter yet; "
            "refusing to emit a metadata-only host result."
        )
    if task != "alfworld" and args.memory_method not in OFFICIAL_RUNTIME_METHODS:
        raise RuntimeError(
            f"{task} has no verified real {args.memory_method!r} memory adapter yet; "
            "refusing to emit a metadata-only memory result."
        )
    from team_memory.integrations import HostRuntimeBridge

    host_bridge = HostRuntimeBridge.build(
        mas=args.mas,
        task=task,
        project_root=Path(os.environ.get("TEAM_MEMORY_PROJECT_ROOT", Path(__file__).resolve().parents[1])),
    )
    host_metadata = host_bridge.metadata()
    if args.mas == "dylan":
        autogen_metadata = HostRuntimeBridge.build(
            mas="autogen",
            task=task,
            project_root=Path(os.environ.get("TEAM_MEMORY_PROJECT_ROOT", Path(__file__).resolve().parents[1])),
        ).metadata()
        if host_metadata["host_topology_hash"] == autogen_metadata["host_topology_hash"]:
            raise RuntimeError("DyLAN host bridge produced the same topology hash as AutoGen")
    memory_execution_mode = f"{task}:{args.memory_method}:official-loop"
    memory_identity = _stable_identity_hash(task, args.memory_method, official_entrypoint)
    memory_extra: dict[str, Any] = {}
    if args.memory_method == "gmemory":
        from team_memory.integrations.gmemory_bridge import ensure_development_snapshot

        project_root = Path(
            os.environ.get("TEAM_MEMORY_PROJECT_ROOT", Path(__file__).resolve().parents[1])
        )
        snapshot = os.environ.get("TEAM_MEMORY_GMEMORY_SNAPSHOT")
        snapshot_path = (
            Path(snapshot)
            if snapshot
            else ensure_development_snapshot(
                project_root=project_root,
                benchmark_task=task,
                evaluation_case_id=getattr(args, "case_id", None),
            )
        )
        manifest_path = snapshot_path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        memory_execution_mode = f"{task}:gmemory:read-only-frozen-snapshot"
        memory_identity = "team_memory.integrations.gmemory_bridge.GMemoryBridge:gmemory-bridge-v1"
        memory_extra = {
            "gmemory_snapshot_id": manifest.get("snapshot_id", ""),
            "gmemory_snapshot_path": str(snapshot_path),
            "gmemory_snapshot_frozen": bool(manifest.get("frozen")),
            "gmemory_snapshot_source_case_ids": manifest.get("source_case_ids", []),
            "gmemory_upstream_components": {
                "memory_class": "external/GMemory/mas/memory/mas_memory/GMemory.py::GMemory",
                "base_class": "external/GMemory/mas/memory/mas_memory/memory_base.py::MASMemoryBase",
                "storage": "langchain_chroma.Chroma persist_directory",
                "message_schema": "external/GMemory/mas/memory/common.py::MASMessage",
                "retrieval_method": "GMemory.retrieve_memory",
                "post_episode_update": "GMemory.add_memory / MASMemoryBase.save_task_context",
            },
        }
    elif args.memory_method == "team-memory":
        memory_execution_mode = f"{memory_execution_mode}:gems-runtime-hooks"
    return {
        **host_metadata,
        "memory_execution_mode": memory_execution_mode,
        "memory_adapter_identity": memory_identity,
        **memory_extra,
        "memory_enabled": args.memory_method != "no-memory",
        "result_schema_version": "e1-runtime-identity-v2",
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_alfworld_case(project_root: Path, case_id: str) -> dict[str, Any]:
    path = project_root / "external/GMemory/data/alfworld/alfworld_tasks_suffix.json"
    tasks = _load_json(path)
    for index, task in enumerate(tasks):
        identifiers = {
            str(index),
            str(task.get("gamefile", "")),
            str(task.get("id", "")),
        }
        if case_id in identifiers:
            return {
                "case_id": str(task.get("gamefile", case_id)),
                "official_task_index": index,
                "task": task,
                "source_path": str(path),
            }
    raise ValueError(f"ALFWorld official case id not found: {case_id!r}")


def _find_webarena_case(project_root: Path, case_id: str) -> dict[str, Any]:
    path = project_root / "external/webarena/config_files/test.raw.json"
    for task in _load_json(path):
        if str(task.get("task_id")) == case_id:
            return {
                "case_id": case_id,
                "official_task_id": task["task_id"],
                "intent": task.get("intent", ""),
                "sites": task.get("sites", []),
                "eval_types": task.get("eval", {}).get("eval_types", []),
                "source_path": str(path),
            }
    raise ValueError(f"WebArena official task_id not found: {case_id!r}")


def _find_multiagentbench_case(project_root: Path, case_id: str) -> dict[str, Any]:
    if "/" not in case_id:
        raise ValueError("MultiAgentBench case_id must be '<scenario>/<task_id_or_line_index>'")
    scenario, raw_id = case_id.split("/", 1)
    path = project_root / f"external/MARBLE/multiagentbench/{scenario}/{scenario}_main.jsonl"
    if not path.is_file():
        raise ValueError(f"MultiAgentBench scenario manifest not found: {scenario!r}")
    with path.open(encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if not line.strip():
                continue
            task = json.loads(line)
            official_id = task.get("task_id", line_index)
            if str(official_id) == raw_id:
                return {
                    "case_id": case_id,
                    "official_task_id": official_id,
                    "official_case_id": raw_id,
                    "scenario": scenario,
                    "agent_count": len(task.get("agents", [])),
                    "task": task,
                    "source_path": str(path),
                    "line_index": line_index,
                }
    raise ValueError(f"MultiAgentBench official case id not found: {case_id!r}")


def _find_officebench_case(project_root: Path, case_id: str) -> dict[str, Any]:
    if "/" not in case_id:
        raise ValueError("OfficeBench case_id must be '<task_dir>/<subtask_id>'")
    task_dir, subtask_id = case_id.split("/", 1)
    path = project_root / f"external/OfficeBench/tasks/{task_dir}/subtasks/{subtask_id}.json"
    if not path.is_file():
        raise ValueError(f"OfficeBench official subtask file not found: {case_id!r}")
    task = _load_json(path)
    return {
        "case_id": case_id,
        "official_task_dir": task_dir,
        "official_subtask_id": subtask_id,
        "instruction": task.get("task", ""),
        "evaluation_functions": [
            item.get("function", "") for item in task.get("evaluation", []) if isinstance(item, dict)
        ],
        "source_path": str(path),
    }


def _resolve_case(project_root: Path, task: str, case_id: str) -> dict[str, Any]:
    if task == "alfworld":
        return _find_alfworld_case(project_root, case_id)
    if task == "webarena":
        return _find_webarena_case(project_root, case_id)
    if task == "multiagentbench":
        return _find_multiagentbench_case(project_root, case_id)
    if task == "officebench":
        return _find_officebench_case(project_root, case_id)
    raise ValueError(f"unsupported task: {task!r}")


def _write_manifest_smoke(
    context: Any,
    *,
    args: argparse.Namespace,
    project_root: Path,
    case: dict[str, Any],
) -> None:
    """Write a standard result after resolving exactly one official case.

    This mode is used for newly downloaded benchmarks until their heavyweight
    browser/Docker/office services are explicitly enabled.  It is still a real
    single-case adapter path: unknown ids fail, and no benchmark list is ever
    iterated as a run.
    """
    if os.environ.get("TEAM_MEMORY_ALLOW_MANIFEST_SMOKE") != "1":
        raise RuntimeError(
            "manifest-smoke mode is not paper evidence. Set "
            "TEAM_MEMORY_ALLOW_MANIFEST_SMOKE=1 only for adapter schema smoke, "
            "or implement the benchmark-specific official runtime harness before "
            "running formal experiments."
        )
    metric_name = {
        "webarena": "success_rate",
        "multiagentbench": "task_score",
        "officebench": "pass_rate",
    }.get(args.task, "primary_score")
    metrics = {
        metric_name: 0.0,
        "primary_score": 0.0,
        "case_count": 1.0,
        "official_case_resolved": 1.0,
    }
    context.write_result(
        metrics,
        cases=[
            {
                **case,
                "case_id": args.case_id,
                "done": False,
                "reward": 0.0,
            }
        ],
        metadata={
            "adapter": "adapters/team_memory_cross_benchmark_adapter.py",
            "execution_mode": "official-single-case-manifest-smoke",
            "official_repository_root": str(project_root),
            "note": (
                "Resolved and materialized exactly one official case. Heavyweight "
                "official runtime is intentionally not started unless a benchmark "
                "specific service harness is enabled."
            ),
        },
    )


def _numeric_score(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        numbers = [_numeric_score(item) for item in value.values()]
        numbers = [item for item in numbers if item >= 0.0]
        return sum(numbers) / len(numbers) if numbers else 0.0
    if isinstance(value, list):
        numbers = [_numeric_score(item) for item in value]
        numbers = [item for item in numbers if item >= 0.0]
        return sum(numbers) / len(numbers) if numbers else 0.0
    return 0.0


def _marble_database_score(task_evaluation: dict[str, Any]) -> float:
    gold_labels = [str(label).strip() for label in task_evaluation.get("root_cause", [])]
    predicted = str(task_evaluation.get("predicted", "")).strip()
    if not gold_labels:
        raise RuntimeError("MARBLE database task_evaluation is missing root_cause labels")
    if not predicted:
        return 0.0
    prompt = (
        f"{predicted}\n\n"
        "From the text above, please identify the two predicted root causes of the issue.\n\n"
        "Please print each of them in the form they appear in two separate lines."
        "I have a very rudimentary system, so if it is not in the exact form, it will crash."
    )
    try:
        from openai import OpenAI

        client = OpenAI(
            base_url=(
                os.environ.get("TEAM_MEMORY_EVAL_ACTOR_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL")
                or os.environ.get("OPENAI_API_BASE")
            ),
            api_key=(
                os.environ.get("TEAM_MEMORY_EVAL_ACTOR_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            ),
            timeout=120,
            max_retries=1,
        )
        response = client.chat.completions.create(
            model=os.environ.get("MARBLE_DATABASE_SCORE_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        content = response.choices[0].message.content or ""
    except Exception as error:
        raise RuntimeError("MARBLE database official score post-processing failed") from error
    predicted_labels = [line.strip() for line in content.splitlines() if line.strip()]
    match_count = sum(gold in predicted_labels for gold in gold_labels)
    return match_count / len(gold_labels)


def _extract_marble_score(payload: dict[str, Any], *, scenario: str) -> tuple[float, str]:
    task_evaluation = payload.get("task_evaluation")
    if scenario in {"database", "db"} and isinstance(task_evaluation, dict):
        return _marble_database_score(task_evaluation), "task_evaluation.database_batch_eval"
    for key in ("final_score", "score", "task_evaluation", "code_quality"):
        if key in payload and payload[key] is not None:
            return _numeric_score(payload[key]), key
    nested = payload.get("evaluator")
    if isinstance(nested, dict):
        for key in ("final_score", "score", "result", "task_evaluation", "code_quality"):
            if key in nested and nested[key] is not None:
                return _numeric_score(nested[key]), f"evaluator.{key}"
    raise RuntimeError(
        "MARBLE official runtime completed but did not write an official score field; "
        f"last official-output keys={sorted(payload.keys())}. Refusing to synthesize primary_score."
    )


def _database_domain_token(task: dict[str, Any]) -> str | None:
    content = str(task.get("task", {}).get("content", "")).upper()
    mapping = {
        "EDUCATION": ("EDUCATIONAL", "STUDENT", "COURSE", "ENROLLMENT"),
        "E_COMMERCE": ("E-COMMERCE", "E COMMERCE", "CUSTOMER", "PRODUCT", "ORDER"),
        "FINANCE": ("FINANCE", "BANK", "TRANSACTION"),
        "HEALTHCARE": ("HEALTHCARE", "PATIENT", "MEDICAL"),
        "MANUFACTURING": ("MANUFACTURING", "RAW MATERIAL", "SUPPLIER"),
        "MUSIC_STREAMING": ("MUSIC", "STREAMING", "PLAYLIST"),
        "FILE_SHARING": ("FILE SHARING", "FILE-SHARING", "FILES"),
        "INTERNET_OF_THINGS": ("INTERNET OF THINGS", "IOT", "DEVICE", "SENSOR"),
    }
    for domain, needles in mapping.items():
        if any(needle in content for needle in needles):
            return domain
    return None


def _database_config_candidates(marble_root: Path, task: dict[str, Any]) -> list[Path]:
    root_causes = [
        str(item).upper()
        for item in task.get("task", {}).get("root_causes", [])
        if str(item).strip()
    ]
    domain = _database_domain_token(task)
    config_dir = marble_root / "marble/configs/test_config_database"
    candidates = sorted(config_dir.glob("*.yaml"))
    if domain:
        candidates = [path for path in candidates if domain in path.stem.upper()]
    if root_causes:
        exact = [
            path
            for path in candidates
            if all(cause in path.stem.upper() for cause in root_causes)
        ]
        if exact:
            return exact
    return candidates


def _fill_marble_research_defaults(data: dict[str, Any]) -> dict[str, Any]:
    """Apply the official multiagentbench/runjsonl2yaml.sh research defaults."""
    selected = json.loads(json.dumps(data))
    defaults = {
        "coordinate_mode": "graph",
        "environment": {
            "max_iterations": 5,
            "name": "Research Collaboration Environment",
            "type": "Research",
        },
        "llm": "gpt-3.5-turbo",
        "memory": {"type": "BaseMemory"},
        "metrics": {"evaluate_llm": "gpt-4o"},
        "output": {"file_path": "result/discussion_output.jsonl"},
    }
    if selected.get("coordinate_mode") == "":
        selected["coordinate_mode"] = defaults["coordinate_mode"]
    if selected.get("llm") == "":
        selected["llm"] = defaults["llm"]
    for key in ("environment", "memory", "output"):
        if isinstance(selected.get(key), dict):
            for sub_key, default_value in defaults[key].items():
                if selected[key].get(sub_key) == "":
                    selected[key][sub_key] = default_value
    if isinstance(selected.get("metrics"), dict) and selected["metrics"].get("evaluate_llm") == "":
        selected["metrics"]["evaluate_llm"] = defaults["metrics"]["evaluate_llm"]
    return selected


def _alfworld_admissible_actions(environment: Any, limit: int = 80) -> list[str]:
    info = getattr(environment, "last_info", {}) or {}
    commands = info.get("admissible_commands") if isinstance(info, dict) else None
    if isinstance(commands, list) and commands:
        first = commands[0]
        if isinstance(first, list):
            return [str(item) for item in first[:limit]]
        return [str(item) for item in commands[:limit]]
    return []


def _parse_alfworld_actor_action(raw_response: str, admissible_actions: list[str]) -> tuple[str, bool]:
    raw = (raw_response or "").strip()
    if not raw:
        return "", True
    candidates: list[str] = []
    for line in raw.splitlines():
        line = line.strip().strip("`").strip()
        if not line:
            continue
        line = re.sub(r"^(?:Action|Command)\s*\d*\s*:\s*", "", line, flags=re.IGNORECASE)
        line = re.sub(r"^\d+[\).\s-]+", "", line).strip()
        line = line.strip("\"'")
        if line:
            candidates.append(line)
    if not candidates:
        return "look", True
    normalized = {item.lower(): item for item in admissible_actions}
    for candidate in candidates:
        exact = normalized.get(candidate.lower())
        if exact is not None:
            return exact, False
    # Preserve the model's first action-like line instead of silently collapsing
    # every parse miss to "look"; the environment will provide official feedback.
    return candidates[0], True


def _alfworld_task_terms(task: str) -> set[str]:
    terms = {item.lower() for item in re.findall(r"[A-Za-z]+", task or "") if len(item) >= 3}
    aliases = {
        "clean": {"clean", "sinkbasin"},
        "heat": {"heat", "microwave"},
        "cool": {"cool", "fridge"},
        "put": {"put", "place"},
        "place": {"put", "place"},
        "examine": {"examine", "look", "desklamp"},
        "light": {"examine", "look", "desklamp"},
    }
    expanded = set(terms)
    for term in terms:
        expanded.update(aliases.get(term, set()))
    return expanded


def _recover_alfworld_action(admissible_actions: list[str], task: str) -> str:
    non_look = [item for item in admissible_actions if item not in {"look"}]
    if not non_look:
        return "inventory" if "inventory" in admissible_actions else "look"
    terms = _alfworld_task_terms(task)
    priority_verbs = (
        "put ",
        "clean ",
        "heat ",
        "cool ",
        "take ",
        "open ",
        "close ",
        "examine ",
        "go to ",
        "inventory",
    )

    def score(action: str) -> tuple[int, int]:
        lower = action.lower()
        term_hits = sum(1 for term in terms if term in lower)
        verb_score = 0
        for index, verb in enumerate(priority_verbs):
            if lower.startswith(verb):
                verb_score = len(priority_verbs) - index
                break
        if lower == "inventory":
            verb_score = 1
        return (term_hits, verb_score)

    return max(non_look, key=score)


def _run_subprocess(
    command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path, timeout: int
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"cwd={cwd}\n")
        log.write(f"command={json.dumps(command, ensure_ascii=False)}\n")
        log.flush()
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            log.write(f"\nofficial runtime timed out after {timeout} seconds\n")
            log.flush()
            raise RuntimeError(
                f"official runtime timed out after {timeout} seconds; see {log_path}"
            ) from exc
    if completed.returncode != 0:
        raise RuntimeError(
            f"official runtime exited with code {completed.returncode}; see {log_path}"
        )


def _positive_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _marble_progress_snapshot(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {"last_iteration": None, "last_agent": None, "last_action": None}
    last_iteration = None
    last_agent = None
    last_action = None
    patterns = (
        re.compile(r"Starting iteration\s+(\d+)"),
        re.compile(r"Agent '([^']+)' is planning the next task"),
        re.compile(r"Agent '([^']+)' acting on task '(.*)'"),
        re.compile(r"Agent '([^']+)' acted with result '(.*)'"),
    )
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-800:]
    for line in lines:
        match = patterns[0].search(line)
        if match:
            last_iteration = int(match.group(1))
        match = patterns[1].search(line)
        if match:
            last_agent = match.group(1)
            last_action = "planning"
        match = patterns[2].search(line)
        if match:
            last_agent = match.group(1)
            last_action = f"acting on task {match.group(2)!r}"
        match = patterns[3].search(line)
        if match:
            last_agent = match.group(1)
            last_action = f"acted with result {match.group(2)[:200]!r}"
    return {
        "last_iteration": last_iteration,
        "last_agent": last_agent,
        "last_action": last_action,
    }


def _runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    project_root = Path(__file__).resolve().parents[1]
    src_path = str(project_root / "src")
    env["TEAM_MEMORY_PROJECT_ROOT"] = str(project_root)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        src_path if not existing_pythonpath else f"{src_path}{os.pathsep}{existing_pythonpath}"
    )
    actor_base = env.get("TEAM_MEMORY_EVAL_ACTOR_BASE_URL")
    actor_key = env.get("TEAM_MEMORY_EVAL_ACTOR_API_KEY")
    if actor_base:
        env["OPENAI_BASE_URL"] = actor_base
        env["OPENAI_API_BASE"] = actor_base
        env["LITELLM_API_BASE"] = actor_base
    if actor_key:
        env["OPENAI_API_KEY"] = actor_key
    env.setdefault("NO_PROXY", "127.0.0.1,localhost")
    env.setdefault("no_proxy", "127.0.0.1,localhost")
    return env


def _read_runtime_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Team Memory runtime metrics sidecar is not an object: {path}")
    return {str(key): float(value) for key, value in payload.items()}


def _configure_gmemory_bridge_env(
    env: dict[str, str],
    *,
    args: argparse.Namespace,
    project_root: Path,
    run_dir: Path,
) -> None:
    if args.memory_method != "gmemory":
        return
    from team_memory.integrations.gmemory_bridge import ensure_development_snapshot

    snapshot = os.environ.get("TEAM_MEMORY_GMEMORY_SNAPSHOT")
    if not snapshot:
        snapshot = str(
            ensure_development_snapshot(
                project_root=project_root,
                benchmark_task=args.task,
                evaluation_case_id=args.case_id,
            ).resolve()
        )
    env["TEAM_MEMORY_GMEMORY_SNAPSHOT"] = snapshot
    env["TEAM_MEMORY_GMEMORY_EVIDENCE"] = str((run_dir / "gmemory-retrieval-evidence.json").resolve())


def _configure_host_trace_env(env: dict[str, str], *, run_dir: Path) -> Path:
    trace_path = run_dir / "host-routing-trace.json"
    env["TEAM_MEMORY_HOST_TRACE"] = str(trace_path.resolve())
    return trace_path


def _assert_gmemory_bridge_invoked(args: argparse.Namespace, metrics: dict[str, float], log_path: Path) -> None:
    if args.memory_method != "gmemory":
        return
    if metrics.get("gmemory_retrieval_invocations", 0.0) <= 0.0:
        raise RuntimeError(
            f"G-Memory official result cannot be accepted because the bridge was never invoked; see {log_path}"
        )
    if metrics.get("gmemory_retrieval_count", 0.0) <= 0.0:
        raise RuntimeError(
            f"G-Memory smoke produced zero retrievals from the frozen snapshot; see {log_path}"
        )


def _run_webarena_official(
    context: Any, *, args: argparse.Namespace, project_root: Path, case: dict[str, Any]
) -> None:
    if args.memory_method not in OFFICIAL_RUNTIME_METHODS:
        raise RuntimeError(
            f"WebArena official runtime is currently wired only for {sorted(OFFICIAL_RUNTIME_METHODS)}; "
            f"{args.memory_method!r} has no real injected runtime yet."
        )
    identity = _runtime_identity_metadata(
        args=args,
        task="webarena",
        official_entrypoint="external/webarena/run.py",
    )
    task_id = int(case["official_task_id"])
    webarena_root = project_root / "external/webarena"
    raw_path = webarena_root / "config_files/test.raw.json"
    raw = raw_path.read_text(encoding="utf-8")
    replacements = {
        "__GITLAB__": os.environ.get("GITLAB", os.environ.get("WEBARENA_GITLAB", "")),
        "__REDDIT__": os.environ.get("REDDIT", os.environ.get("WEBARENA_REDDIT", "")),
        "__SHOPPING__": os.environ.get("SHOPPING", os.environ.get("WEBARENA_SHOPPING", "")),
        "__SHOPPING_ADMIN__": os.environ.get(
            "SHOPPING_ADMIN", os.environ.get("WEBARENA_SHOPPING_ADMIN", "")
        ),
        "__WIKIPEDIA__": os.environ.get("WIKIPEDIA", os.environ.get("WEBARENA_WIKIPEDIA", "")),
        "__MAP__": os.environ.get("MAP", os.environ.get("WEBARENA_MAP", "")),
    }
    missing = [key for key, value in replacements.items() if key in raw and not value]
    if missing:
        raise RuntimeError(
            f"WebArena official runtime requires site URL env vars for {missing}"
        )
    for key, value in replacements.items():
        raw = raw.replace(key, value)
    data = json.loads(raw)
    if task_id >= len(data):
        raise RuntimeError(f"WebArena task_id outside official manifest: {task_id}")
    config_file = webarena_root / f"config_files/{task_id}.json"
    config_file.write_text(json.dumps(data[task_id], ensure_ascii=False, indent=2), encoding="utf-8")
    run_dir = context.output_path.parent / f"{context.output_path.stem}.webarena"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    log_path = run_dir / "official-runtime.log"
    runtime_metrics_path = run_dir / "team-memory-runtime-metrics.json"
    host_trace_path = run_dir / "host-routing-trace.json"
    env = _runtime_env()
    env["GITLAB"] = replacements["__GITLAB__"]
    env["REDDIT"] = replacements["__REDDIT__"]
    env["SHOPPING"] = replacements["__SHOPPING__"]
    env["SHOPPING_ADMIN"] = replacements["__SHOPPING_ADMIN__"]
    env["WIKIPEDIA"] = replacements["__WIKIPEDIA__"]
    env["MAP"] = replacements["__MAP__"]
    if not env.get("HOMEPAGE"):
        homepage = os.environ.get("WEBARENA_HOMEPAGE", "")
        if not homepage:
            parsed = urlparse(replacements["__SHOPPING__"])
            if parsed.scheme and parsed.hostname:
                homepage = f"{parsed.scheme}://{parsed.hostname}:4399"
        env["HOMEPAGE"] = homepage
    env["TEAM_MEMORY_RUNTIME_METRICS"] = str(runtime_metrics_path.resolve())
    env.setdefault("WEBARENA_PLAYWRIGHT_TIMEOUT_MS", "90000")
    host_trace_path = _configure_host_trace_env(env, run_dir=run_dir)
    _configure_gmemory_bridge_env(env, args=args, project_root=project_root, run_dir=run_dir)
    webarena_python = os.environ.get("WEBARENA_PYTHON", sys.executable)
    webarena_python_dir = str(Path(webarena_python).resolve().parent)
    env["PATH"] = f"{webarena_python_dir}{os.pathsep}{env.get('PATH', '')}"
    env["PYTHONPATH"] = f"{str(webarena_root)}{os.pathsep}{env.get('PYTHONPATH', '')}"
    command = [
        webarena_python,
        "run.py",
        "--provider",
        "openai",
        "--model",
        args.actor_model,
        "--mode",
        "chat",
        "--temperature",
        "0",
        "--instruction_path",
        "agent/prompts/jsons/p_cot_id_actree_2s.json",
        "--test_start_idx",
        str(task_id),
        "--test_end_idx",
        str(task_id + 1),
        "--result_dir",
        str(run_dir),
    ]
    _run_subprocess(command, cwd=webarena_root, env=env, log_path=log_path, timeout=7200)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"\[Result\]\s+\((PASS|FAIL)\)", text)
    if not match:
        match = re.search(r"Average score:\s*([0-9.]+)", text)
        if not match:
            raise RuntimeError(
                f"WebArena official runtime did not emit a task score; see {log_path}"
            )
        score = float(match.group(1))
    else:
        score = 1.0 if match.group(1) == "PASS" else 0.0
    official_metrics = {
            "success_rate": score,
            "primary_score": score,
            "case_count": 1.0,
        }
    metrics = {**official_metrics, **_read_runtime_metrics(runtime_metrics_path)}
    _assert_gmemory_bridge_invoked(args, metrics, log_path)
    context.write_result(
        metrics,
        cases=[{**case, "case_id": args.case_id, "reward": score, "done": bool(score)}],
        metadata={
            "adapter": "adapters/team_memory_cross_benchmark_adapter.py",
            "execution_mode": "official-webarena-single-case",
            "official_entrypoint": "external/webarena/run.py",
            "official_log": str(log_path),
            "team_memory_injected": bool(metrics.get("team_memory_enabled", 0.0)),
            "team_memory_runtime_metrics": str(runtime_metrics_path),
            "host_trace": str(host_trace_path),
            **identity,
        },
    )


def _run_multiagentbench_official(
    context: Any, *, args: argparse.Namespace, project_root: Path, case: dict[str, Any]
) -> None:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on external benchmark env.
        raise RuntimeError(
            "MARBLE official runtime requires PyYAML. Install the MARBLE/benchmark "
            "environment before running MultiAgentBench official cells."
        ) from exc

    if args.memory_method not in OFFICIAL_RUNTIME_METHODS:
        raise RuntimeError(
            f"MARBLE official runtime is currently wired only for {sorted(OFFICIAL_RUNTIME_METHODS)}; "
            f"{args.memory_method!r} has no real injected runtime yet."
        )
    identity = _runtime_identity_metadata(
        args=args,
        task="multiagentbench",
        official_entrypoint="external/MARBLE/marble/main.py",
    )
    marble_root = project_root / "external/MARBLE"
    scenario = str(case.get("scenario", args.case_id.split("/", 1)[0])).lower()
    official_id = str(case.get("official_case_id", args.case_id).split("/", 1)[-1])
    config_candidates: list[Path] = []
    if scenario == "coding":
        config_candidates.append(
            marble_root / "marble/configs/coding_configs" / f"config_{official_id}.yaml"
        )
    elif scenario == "minecraft":
        config_candidates.extend(
            sorted((marble_root / "marble/configs/test_config_minecraft").glob(f"*_{official_id}.yaml"))
        )
    elif scenario == "research":
        config_candidates.extend(
            sorted((marble_root / "marble/configs/test_config_research").glob(f"*{official_id}*.yaml"))
        )
    elif scenario in {"database", "db"}:
        config_candidates.extend(_database_config_candidates(marble_root, case.get("task", {})))
    official_config = next((path for path in config_candidates if path.exists()), None)
    official_config_source = str(official_config) if official_config else str(case.get("source_path", ""))
    if official_config is None and scenario == "research" and isinstance(case.get("task"), dict):
        selected = _fill_marble_research_defaults(case["task"])
        official_config_source = f"{case.get('source_path')}#line-{case.get('line_index')}"
    elif official_config is None:
        raise RuntimeError(
            f"MARBLE official single-case config not found for {args.case_id!r}; "
            "refusing to synthesize a non-official config."
        )
    else:
        selected = yaml.safe_load(official_config.read_text(encoding="utf-8"))
    run_dir = context.output_path.parent / f"{context.output_path.stem}.marble"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    output_file = run_dir / "official-output.jsonl"
    config_file = run_dir / "single-case-config.yaml"
    selected["llm"] = args.actor_model
    selected["output"] = {"file_path": str(output_file.resolve()), "format": "jsonl"}
    config_file.write_text(yaml.safe_dump(selected, allow_unicode=True, sort_keys=False), encoding="utf-8")
    env = _runtime_env()
    env["PYTHONPATH"] = f"{marble_root}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env.setdefault("LITELLM_LOG", "ERROR")
    marble_timeout = _positive_int_env("TEAM_MEMORY_MARBLE_TIMEOUT_SECONDS", 7200)
    marble_llm_timeout = _positive_int_env("TEAM_MEMORY_MARBLE_LLM_TIMEOUT_SECONDS", 120)
    env["TEAM_MEMORY_MARBLE_LLM_TIMEOUT_SECONDS"] = str(marble_llm_timeout)
    env.setdefault("LITELLM_REQUEST_TIMEOUT", str(marble_llm_timeout))
    (marble_root / "marble/logs").mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "official-runtime.log"
    runtime_metrics_path = run_dir / "team-memory-runtime-metrics.json"
    env["TEAM_MEMORY_RUNTIME_METRICS"] = str(runtime_metrics_path.resolve())
    host_trace_path = _configure_host_trace_env(env, run_dir=run_dir)
    _configure_gmemory_bridge_env(env, args=args, project_root=project_root, run_dir=run_dir)
    try:
        _run_subprocess(
            [sys.executable, "main.py", "--config_path", str(config_file.resolve())],
            cwd=marble_root / "marble",
            env=env,
            log_path=log_path,
            timeout=marble_timeout,
        )
    except RuntimeError as exc:
        if "timed out after" in str(exc):
            snapshot = _marble_progress_snapshot(log_path)
            error_path = run_dir / "error.txt"
            error_path.write_text(
                json.dumps(
                    {
                        "error": str(exc),
                        "timeout_seconds": marble_timeout,
                        "llm_timeout_seconds": marble_llm_timeout,
                        "last_progress": snapshot,
                        "official_log": str(log_path),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            raise RuntimeError(
                f"{exc}; last MARBLE progress: {json.dumps(snapshot, ensure_ascii=False)}"
            ) from exc
        raise
    if not output_file.exists():
        raise RuntimeError(f"MARBLE official runtime did not write {output_file}")
    rows = [
        json.loads(line)
        for line in output_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    payload = rows[-1] if rows else {}
    score, score_source = _extract_marble_score(payload, scenario=scenario)
    official_metrics = {
            "task_score": score,
            "primary_score": score,
            "case_count": 1.0,
            "marble_official_score_present": 1.0,
            "marble_token_usage": _numeric_score(payload.get("token_usage", 0.0)),
        }
    metrics = {**official_metrics, **_read_runtime_metrics(runtime_metrics_path)}
    _assert_gmemory_bridge_invoked(args, metrics, log_path)
    context.write_result(
        metrics,
        cases=[{**case, "case_id": args.case_id, "reward": score, "done": score > 0.0}],
        metadata={
            "adapter": "adapters/team_memory_cross_benchmark_adapter.py",
            "execution_mode": "official-marble-single-case",
            "official_entrypoint": "external/MARBLE/marble/main.py",
            "official_config": official_config_source,
            "official_output": str(output_file),
            "official_log": str(log_path),
            "official_score_source": score_source,
            "team_memory_injected": bool(metrics.get("team_memory_enabled", 0.0)),
            "team_memory_runtime_metrics": str(runtime_metrics_path),
            "host_trace": str(host_trace_path),
            **identity,
        },
    )


def _run_officebench_official(
    context: Any, *, args: argparse.Namespace, project_root: Path, case: dict[str, Any]
) -> None:
    if args.memory_method not in OFFICIAL_RUNTIME_METHODS:
        raise RuntimeError(
            f"OfficeBench official runtime is currently wired only for {sorted(OFFICIAL_RUNTIME_METHODS)}; "
            f"{args.memory_method!r} has no real injected runtime yet."
        )
    identity = _runtime_identity_metadata(
        args=args,
        task="officebench",
        official_entrypoint="external/OfficeBench/agent_interact.py",
    )
    office_root = project_root / "external/OfficeBench"
    task_dir = f"tasks/{case['official_task_dir']}"
    subtask_id = str(case["official_subtask_id"])
    tag = f"tm-{context.output_path.stem}"
    output_dir = (
        office_root
        / task_dir
        / "outputs"
        / subtask_id
        / f"{args.actor_model.replace('/', '_')}_{tag}"
    )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    env = _runtime_env()
    env["PYTHONPATH"] = str(office_root)
    log_path = context.output_path.parent / f"{context.output_path.stem}.officebench.log"
    runtime_metrics_path = context.output_path.parent / f"{context.output_path.stem}.team-memory-runtime-metrics.json"
    env["TEAM_MEMORY_RUNTIME_METRICS"] = str(runtime_metrics_path.resolve())
    host_trace_path = _configure_host_trace_env(env, run_dir=context.output_path.parent)
    _configure_gmemory_bridge_env(
        env,
        args=args,
        project_root=project_root,
        run_dir=context.output_path.parent,
    )
    command = [
        sys.executable,
        "agent_interact.py",
        "--model_name",
        args.actor_model,
        "--task_dir",
        task_dir,
        "--config_file",
        f"{task_dir}/subtasks/{subtask_id}.json",
        "--tag",
        tag,
        "--mode",
        "force_new",
    ]
    _run_subprocess(command, cwd=office_root, env=env, log_path=log_path, timeout=7200)
    sys.path.insert(0, str(office_root))
    previous = Path.cwd()
    os.chdir(office_root)
    try:
        from evaluation import evaluate_output

        passed = bool(evaluate_output(case["official_task_dir"], subtask_id, str(output_dir / "testbed")))
    finally:
        os.chdir(previous)
    score = 1.0 if passed else 0.0
    official_metrics = {
            "pass_rate": score,
            "primary_score": score,
            "case_count": 1.0,
        }
    metrics = {**official_metrics, **_read_runtime_metrics(runtime_metrics_path)}
    _assert_gmemory_bridge_invoked(args, metrics, log_path)
    context.write_result(
        metrics,
        cases=[{**case, "case_id": args.case_id, "reward": score, "done": passed}],
        metadata={
            "adapter": "adapters/team_memory_cross_benchmark_adapter.py",
            "execution_mode": "official-officebench-single-case",
            "official_entrypoint": "external/OfficeBench/agent_interact.py",
            "official_output_dir": str(output_dir),
            "official_log": str(log_path),
            "team_memory_injected": bool(metrics.get("team_memory_enabled", 0.0)),
            "team_memory_runtime_metrics": str(runtime_metrics_path),
            "host_trace": str(host_trace_path),
            **identity,
        },
    )


def _run_alfworld_team_memory_official(
    context: Any, *, args: argparse.Namespace, project_root: Path, case: dict[str, Any]
) -> None:
    if os.environ.get("TEAM_MEMORY_ALFWORLD_USE_GMEMORY_OFFICIAL_MAS", "1") == "1":
        return _delegate_alfworld_to_gmemory(args, project_root)
    if args.memory_method != "team-memory":
        return _delegate_alfworld_to_gmemory(args, project_root)
    gmemory_root = project_root / "external/GMemory"
    sys.path.insert(0, str(project_root / "src"))
    sys.path.insert(0, str(gmemory_root / "tasks"))
    sys.path.insert(0, str(gmemory_root))
    previous = Path.cwd()
    os.chdir(gmemory_root)
    try:
        import yaml
        from openai import OpenAI
        from envs.alfworld_env import AlfworldEnv, get_env_name_from_gamefile
        from team_memory.benchmark_runtime import TeamMemoryBenchmarkRuntime
        import textworld.envs.pddl.textgen as textgen

        def _derive_with_explicit_eval_locals(self, context=None):
            context = context or self.context
            value = eval(self.expression, {}, context["variables"])
            return [textgen.TerminalSymbol(value)]

        textgen.EvalSymbol.derive = _derive_with_explicit_eval_locals

        config_path = gmemory_root / "tasks/env_configs/alfworld_config.yaml"
        env_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        max_steps = int(os.environ.get("TEAM_MEMORY_ALFWORLD_SINGLE_CASE_MAX_STEPS", "30"))
        environment = AlfworldEnv(env_config, max_trials=max_steps)
        raw_task_config = dict(case["task"])
        gamefile = raw_task_config.get("gamefile") or case.get("case_id")
        env_name = get_env_name_from_gamefile(str(gamefile))
        task_config = {
            "task": raw_task_config.get("goal", raw_task_config.get("task", "")),
            "env_kwargs": {"config": "alfworld", "gamefile": gamefile},
            "env_name": env_name,
        }
        task_main, task_description = environment.set_env(task_config)
    finally:
        os.chdir(previous)
    runtime = TeamMemoryBenchmarkRuntime.from_environment(
        main_goal=task_main,
        plan_actions=[
            "read ALFWorld observation",
            f"{args.mas} agent proposes next admissible text action",
            "apply action in official ALFWorld environment",
            "evaluate official reward/done",
        ],
        plan_edges=[(0, 1), (1, 2), (2, 3)],
    )
    payload = runtime.start_agent(
        f"alfworld-{args.mas}-agent",
        f"{args.mas} ALFWorld task-solving agent",
        current_task=task_main,
        task_query=task_description,
        sop_limit=1,
    )
    prompt_prefix = payload.get("prompt_prefix") or ""
    observation = task_description
    runtime.observe(
        f"alfworld-{args.mas}-agent",
        observation,
        task=task_main,
    )
    trajectory: list[dict[str, Any]] = []
    actor_base_url = os.environ.get("TEAM_MEMORY_EVAL_ACTOR_BASE_URL")
    actor_api_key = os.environ.get("TEAM_MEMORY_EVAL_ACTOR_API_KEY")
    client = (
        OpenAI(base_url=actor_base_url, api_key=actor_api_key, timeout=60, max_retries=1)
        if actor_base_url and actor_api_key
        else None
    )
    done = False
    reward = 0.0
    parse_error_count = 0
    empty_response_count = 0
    parser_recovery_count = 0
    prompt_char_counts: list[int] = []
    state_conflict_candidate_count = 0
    for step in range(max_steps):
        admissible_actions = _alfworld_admissible_actions(environment)
        recent = trajectory[-3:]
        recent_text = "\n".join(
            f"- step {item['step']}: action={item['action']!r}, reward={item['reward']}, "
            f"done={item['done']}, observation={str(item['observation'])[:500]}"
            for item in recent
        ) or "None yet."
        admissible_text = "\n".join(f"- {item}" for item in admissible_actions[:60])
        admissible_block = admissible_text or "- look\n- inventory"
        base_prompt = (
            f"{prompt_prefix}\n\n" if prompt_prefix else ""
        ) + (
            f"Official ALFWorld task:\n{task_main}\n\n"
            f"Current observation:\n{observation}\n\n"
            f"Recent action feedback:\n{recent_text}\n\n"
            f"Current admissible actions:\n{admissible_block}\n\n"
            "Choose one useful action from the current admissible actions. Prefer actions "
            "that make progress toward the task over repeating 'look'. You may use "
            "'inventory' when you need to check what you are carrying.\n"
            "Return exactly one ALFWorld text action and nothing else."
        )
        prompt = base_prompt
        prompt_char_counts.append(len(prompt))
        action = "look"
        raw_action_response = ""
        parse_error = False
        actor_error = ""
        parser_recovery_used = False

        def _completion(action_prompt: str, *, recovery: bool = False) -> str:
            if client is None:
                return ""
            completion_kwargs: dict[str, Any] = {
                "model": args.actor_model,
                "messages": [{"role": "user", "content": action_prompt}],
            }
            if args.actor_model.startswith("gpt-5"):
                completion_kwargs["temperature"] = 1.0
                completion_kwargs["max_completion_tokens"] = 128 if recovery else 256
            else:
                completion_kwargs["temperature"] = 0.0
                completion_kwargs["max_tokens"] = 32 if recovery else 64
            response = client.chat.completions.create(**completion_kwargs)
            return response.choices[0].message.content or ""

        try:
            if client is not None:
                raw_action_response = _completion(prompt)
                if not raw_action_response.strip():
                    empty_response_count += 1
                    recovery_prompt = (
                        f"Official ALFWorld task:\n{task_main}\n\n"
                        f"Current observation:\n{observation}\n\n"
                        f"Current admissible actions:\n{admissible_block}\n\n"
                        "Choose exactly one command from the admissible actions above. "
                        "Return only the command text."
                    )
                    prompt_char_counts.append(len(recovery_prompt))
                    raw_action_response = _completion(recovery_prompt, recovery=True)
                    if not raw_action_response.strip():
                        empty_response_count += 1
                action, parse_error = _parse_alfworld_actor_action(
                    raw_action_response,
                    admissible_actions,
                )
                if not action:
                    action = _recover_alfworld_action(admissible_actions, task_main)
                    parser_recovery_count += 1
                    parser_recovery_used = True
                    parse_error = True
                if parse_error:
                    parse_error_count += 1
                    runtime.record_error(
                        f"alfworld-{args.mas}-agent",
                        f"action_parse_error: raw={raw_action_response!r}; parsed={action!r}",
                        task=task_main,
                    )
        except Exception as error:
            actor_error = str(error)
            runtime.record_error(f"alfworld-{args.mas}-agent", error, task=task_main)
            parse_error_count += 1
            parse_error = True
            empty_response_count += 1
            action = _recover_alfworld_action(admissible_actions, task_main)
            parser_recovery_count += 1
            parser_recovery_used = True
        runtime.before_action(
            f"alfworld-{args.mas}-agent",
            action,
            task=task_main,
        )
        try:
            observation, reward, done = environment.step(action)
        except Exception as error:
            runtime.record_error(f"alfworld-{args.mas}-agent", error, task=task_main)
            observation = str(error)
            reward = 0.0
            done = False
            break
        runtime.after_action(
            f"alfworld-{args.mas}-agent",
            observation,
            task=task_main,
            state_key="alfworld_task_done" if done else None,
            state_value=True if done else None,
            authoritative=bool(done),
        )
        if done and runtime.enabled:
            state_conflict_candidate_count += 1
        trajectory.append(
            {
                "step": step + 1,
                "action": action,
                "raw_action_response": raw_action_response,
                "parse_error": parse_error,
                "parser_recovery_used": parser_recovery_used,
                "actor_error": actor_error,
                "admissible_actions": admissible_actions,
                "observation": observation,
                "reward": reward,
                "done": done,
            }
        )
        if done:
            break
    official_reward, success, message = environment.feedback()
    metrics = runtime.finish_metrics(
        {
            "success_rate": float(official_reward),
            "primary_score": float(official_reward),
            "case_count": 1.0,
            "parse_error_count": float(parse_error_count),
            "non_look_action_count": float(
                sum(1 for item in trajectory if item.get("action") != "look")
            ),
            "trajectory_len": float(len(trajectory)),
            "empty_response_count": float(empty_response_count),
            "parser_recovery_count": float(parser_recovery_count),
            "prompt_char_count_mean": (
                float(sum(prompt_char_counts) / len(prompt_char_counts))
                if prompt_char_counts
                else 0.0
            ),
            "prompt_char_count_max": float(max(prompt_char_counts) if prompt_char_counts else 0),
            "state_conflict_candidate_count": float(state_conflict_candidate_count),
        }
    )
    context.write_result(
        metrics,
        cases=[
            {
                **case,
                "case_id": args.case_id,
                "reward": float(official_reward),
                "done": bool(success),
                "trajectory": trajectory,
            }
        ],
        metadata={
            "adapter": "adapters/team_memory_cross_benchmark_adapter.py",
            "execution_mode": "official-alfworld-single-case-team-memory-loop",
            "official_entrypoint": "external/GMemory/tasks/envs/alfworld_env.py",
            "max_steps": max_steps,
            "max_steps_env": "TEAM_MEMORY_ALFWORLD_SINGLE_CASE_MAX_STEPS",
            "team_memory_injected": True,
            "mas": args.mas,
            **_runtime_identity_metadata(
                args=args,
                task="alfworld",
                official_entrypoint="external/GMemory/tasks/envs/alfworld_env.py",
            ),
            "message": message,
            "prompt_char_count_mean": (
                float(sum(prompt_char_counts) / len(prompt_char_counts))
                if prompt_char_counts
                else 0.0
            ),
            "prompt_char_count_max": float(max(prompt_char_counts) if prompt_char_counts else 0),
        },
    )
    return None


def _delegate_alfworld_to_gmemory(args: argparse.Namespace, project_root: Path) -> int:
    if args.memory_method not in GMEMORY_METHODS:
        raise ValueError(
            f"ALFWorld official GMemory adapter supports {sorted(GMEMORY_METHODS)}, "
            f"not {args.memory_method!r}"
        )
    if args.mas == "agentnet":
        raise ValueError("ALFWorld GMemory adapter has no official AgentNet MAS entrypoint")
    gmemory_root = project_root / "external/GMemory"
    if not (gmemory_root / "tasks/run.py").is_file():
        raise FileNotFoundError(f"missing official GMemory checkout: {gmemory_root}")
    sys.path.insert(0, str(project_root))
    from adapters import team_memory_gmemory_adapter

    previous = Path.cwd()
    os.chdir(gmemory_root)
    try:
        g_args = argparse.Namespace(
            task=args.task,
            mas=args.mas,
            actor_model=args.actor_model,
            sop_model=args.sop_model,
            memory_method=args.memory_method,
            seed=args.seed,
            output=args.output,
        )
        metrics, cases, metadata = team_memory_gmemory_adapter.run(g_args)
    finally:
        os.chdir(previous)

    from team_memory.evaluation_adapter import EvaluationContext

    context = EvaluationContext.from_environment()
    metadata = {
        **metadata,
        "delegated_by": "adapters/team_memory_cross_benchmark_adapter.py",
        "case_id_cli": args.case_id,
    }
    context.write_result(metrics, cases=cases, metadata=metadata)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=SUPPORTED_TASKS)
    parser.add_argument("--mas", required=True)
    parser.add_argument("--actor-model", required=True)
    parser.add_argument("--sop-model", required=True)
    parser.add_argument("--memory-method", required=True, choices=SUPPORTED_MEMORY_METHODS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--ablation", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.case_id:
        raise RuntimeError("--case-id is required; refusing to run a full benchmark sweep")
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from team_memory.evaluation_adapter import EvaluationContext

    context = EvaluationContext.from_environment()
    expected = {
        "task": args.task,
        "mas_framework": args.mas,
        "actor_model": args.actor_model,
        "sop_model": args.sop_model,
        "memory_method": args.memory_method,
        "seed": args.seed,
        "case_id": args.case_id,
        "ablation": args.ablation,
        "output_path": args.output.resolve(),
    }
    actual = {
        "task": context.task,
        "mas_framework": context.mas_framework,
        "actor_model": context.actor_model,
        "sop_model": context.sop_model,
        "memory_method": context.memory_method,
        "seed": context.seed,
        "case_id": context.case_id,
        "ablation": context.ablation,
        "output_path": context.output_path.resolve(),
    }
    if expected != actual:
        raise RuntimeError(f"CLI/environment experiment identity mismatch: {expected} != {actual}")

    case = _resolve_case(project_root, args.task, args.case_id)
    if os.environ.get("TEAM_MEMORY_ALLOW_MANIFEST_SMOKE") == "1" and args.task != "alfworld":
        _write_manifest_smoke(context, args=args, project_root=project_root, case=case)
        return 0
    if args.task == "alfworld":
        if args.memory_method == "team-memory":
            _run_alfworld_team_memory_official(context, args=args, project_root=project_root, case=case)
            return 0
        return _delegate_alfworld_to_gmemory(args, project_root)
    if args.task == "webarena":
        _run_webarena_official(context, args=args, project_root=project_root, case=case)
    elif args.task == "multiagentbench":
        _run_multiagentbench_official(context, args=args, project_root=project_root, case=case)
    elif args.task == "officebench":
        _run_officebench_official(context, args=args, project_root=project_root, case=case)
    else:
        raise RuntimeError(f"no official runtime harness configured for task {args.task!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
