"""统一展开、执行和恢复最小贡献实验矩阵。

公开 benchmark 的依赖和生命周期不同，因此外部单元通过命令适配器运行。适配器必须把
标准 JSON 写到 ``TEAM_MEMORY_EVAL_OUTPUT``。本 runner 负责四个贡献条件、密钥注入、case
级状态、日志和断点续跑；一个 case 失败不会导致已完成 case 重新执行。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import random
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .benchmark import run_all_benchmarks


SUCCESS = "success"
CONTRIBUTION_CONDITIONS = (
    "no-extra-components",
    "blackboard-only",
    "sop-only",
    "divergence-only",
    "full",
)


def load_env_file(path: str | Path, *, override: bool = False) -> None:
    """安全解析简单 KEY=VALUE 文件；不执行 shell、不展开命令。"""
    source = Path(path)
    if not source.exists():
        return
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{source}:{line_number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"{source}:{line_number}: invalid environment variable name")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value


def load_matrix(path: str | Path) -> dict[str, Any]:
    """读取单一评测文件并检查跨表引用和密钥隔离。"""
    source = Path(path)
    config = json.loads(source.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "output_dir",
        "state_db",
        "provider",
        "models",
        "mas_frameworks",
        "memory_methods",
        "benchmarks",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"evaluation matrix missing keys: {sorted(missing)}")
    if config["schema_version"] != 3:
        raise ValueError("unsupported evaluation matrix schema_version")
    provider = config["provider"]
    for name in ("base_url_env", "key_env"):
        if not isinstance(provider.get(name), str):
            raise ValueError(f"provider.{name} is required")
    forbidden = {"api_key", "token", "secret"}
    if forbidden.intersection(provider):
        raise ValueError("literal secrets are forbidden; use provider.key_env")
    tables = {}
    for key in ("models", "mas_frameworks", "memory_methods", "benchmarks"):
        rows = config[key]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{key} must be a non-empty list")
        indexed = {row["id"]: row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError(f"duplicate id in {key}")
        tables[key] = indexed
    for model in config["models"]:
        for name in ("base_url_env", "key_env"):
            if name in model and not isinstance(model[name], str):
                raise ValueError(f"models.{model['id']}.{name} must be a string")
        if forbidden.intersection(model):
            raise ValueError(
                f"literal secrets are forbidden in model {model['id']}; use key_env"
            )
    for benchmark in config["benchmarks"]:
        for method in benchmark["methods"]:
            if method not in tables["memory_methods"]:
                raise ValueError(f"unknown memory method {method!r}")
        tasks = benchmark.get("tasks", ["all"])
        if not isinstance(tasks, list) or not tasks or not all(isinstance(task, str) for task in tasks):
            raise ValueError(f"{benchmark['id']}.tasks must be a non-empty string list")
        case_ids = benchmark.get("case_ids")
        if case_ids is not None:
            if isinstance(case_ids, list):
                if not case_ids or not all(isinstance(item, str) for item in case_ids):
                    raise ValueError(f"{benchmark['id']}.case_ids must be a non-empty string list")
            elif isinstance(case_ids, dict):
                for task, items in case_ids.items():
                    if task not in tasks:
                        raise ValueError(f"{benchmark['id']}.case_ids contains unknown task {task!r}")
                    if not isinstance(items, list) or not items or not all(isinstance(item, str) for item in items):
                        raise ValueError(f"{benchmark['id']}.case_ids.{task} must be a non-empty string list")
            else:
                raise ValueError(f"{benchmark['id']}.case_ids must be a list or object")
        for framework in benchmark["mas_frameworks"]:
            if framework not in tables["mas_frameworks"]:
                raise ValueError(f"unknown MAS framework {framework!r}")
        if benchmark["uses_llm"]:
            actor_models = benchmark.get("actor_models", [])
            if not actor_models:
                raise ValueError(f"{benchmark['id']}.actor_models must not be empty")
            unknown_actor_models = set(actor_models) - set(tables["models"])
            if unknown_actor_models:
                raise ValueError(
                    f"{benchmark['id']} contains unknown actor models: "
                    f"{sorted(unknown_actor_models)}"
                )
            methods_using_sop_agent = [
                method
                for method in benchmark["methods"]
                if tables["memory_methods"][method].get("uses_sop_agent", False)
            ]
            sop_models = benchmark.get("sop_models", [])
            if methods_using_sop_agent and not sop_models:
                raise ValueError(
                    f"{benchmark['id']}.sop_models is required for {methods_using_sop_agent}"
                )
            unknown_sop_models = set(sop_models) - set(tables["models"])
            if unknown_sop_models:
                raise ValueError(
                    f"{benchmark['id']} contains unknown SOP models: "
                    f"{sorted(unknown_sop_models)}"
                )
        for method, frameworks in benchmark.get("method_mas", {}).items():
            if method not in benchmark["methods"]:
                raise ValueError(f"{benchmark['id']}.method_mas contains unknown method {method!r}")
            unknown_frameworks = set(frameworks) - set(benchmark["mas_frameworks"])
            if unknown_frameworks:
                raise ValueError(
                    f"{benchmark['id']}.{method} contains unknown MAS: "
                    f"{sorted(unknown_frameworks)}"
                )
        known_ablations = {row["id"] for row in config.get("ablation_variants", [{"id": "full"}])}
        for variant in benchmark.get("ablations", ["full"]):
            if variant not in known_ablations:
                raise ValueError(f"unknown ablation variant {variant!r}")
        if benchmark.get("paper_role") == "contribution-main":
            if tuple(benchmark.get("ablations", [])) != CONTRIBUTION_CONDITIONS:
                raise ValueError(
                    f"{benchmark['id']} must use the configured minimal contribution conditions"
                )
        if benchmark["runner"] == "external":
            jobs = benchmark.get("method_jobs", {})
            if not isinstance(jobs, dict):
                raise ValueError(f"{benchmark['id']}.method_jobs must be an object")
            unknown_jobs = jobs.keys() - set(benchmark["methods"])
            if unknown_jobs:
                raise ValueError(
                    f"{benchmark['id']} has jobs for methods outside its matrix: "
                    f"{sorted(unknown_jobs)}"
                )
    config["_source"] = str(source.resolve())
    return config


@dataclass(frozen=True)
class EvalCell:
    benchmark: str
    task: str
    memory_method: str
    actor_model: str
    sop_model: str
    mas_framework: str
    seed: int
    ablation: str = "full"
    case_id: str | None = None

    @property
    def key(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
        parts = [
            self.benchmark,
            self.task,
            self.memory_method,
            self.actor_model,
            self.sop_model,
            self.mas_framework,
            self.ablation,
            f"s{self.seed}",
        ]
        if self.case_id:
            parts.append(self.case_id)
        readable = "__".join(re.sub(r"[^A-Za-z0-9_.-]+", "-", part) for part in parts)
        if len(readable) > 180:
            readable = readable[:180].rstrip("-_.")
        return f"{readable}__{digest}"


def _enabled(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in rows if row.get("enabled", True)}


def _case_ids_for_task(
    benchmark: dict[str, Any], task_id: str, forced_case_id: str | None
) -> list[str | None]:
    if forced_case_id:
        return [forced_case_id]
    configured = benchmark.get("case_ids")
    if configured is None:
        return [None]
    if isinstance(configured, list):
        return list(configured)
    return list(configured.get(task_id, [None]))


def _supported_by_benchmark_cell_rules(
    benchmark: dict[str, Any], task_id: str, method_id: str, framework_id: str
) -> bool:
    """Apply optional task-level support rules for partially wired external suites."""
    rules = benchmark.get("supported_cells")
    if not rules:
        return True
    if not isinstance(rules, list):
        raise ValueError(f"{benchmark['id']}.supported_cells must be a list")
    for rule in rules:
        if not isinstance(rule, dict):
            raise ValueError(f"{benchmark['id']}.supported_cells entries must be objects")
        tasks = rule.get("tasks", ["*"])
        methods = rule.get("methods", ["*"])
        frameworks = rule.get("mas_frameworks", ["*"])
        if (
            ("*" in tasks or task_id in tasks)
            and ("*" in methods or method_id in methods)
            and ("*" in frameworks or framework_id in frameworks)
        ):
            return True
    return False


def expand_matrix(
    config: dict[str, Any],
    *,
    benchmark_filter: str | None = None,
    task_filter: str | None = None,
    method_filter: str | None = None,
    actor_model_filter: str | None = None,
    sop_model_filter: str | None = None,
    mas_filter: str | None = None,
    ablation_filter: str | None = None,
    seed_filter: int | None = None,
    case_id: str | None = None,
) -> list[EvalCell]:
    """确定性展开公平实验单元。

    Actor 模型负责完成 benchmark 任务，SOP 模型只负责 Team Memory 的候选整理与验证。
    两个角色必须分列，否则“更好的记忆”会与“更强的任务模型”混成同一个变量。若 benchmark
    声明 ``case_ids``，每个 case 都会成为独立 checkpoint 单元。
    """
    models = _enabled(config["models"])
    frameworks = _enabled(config["mas_frameworks"])
    methods = _enabled(config["memory_methods"])
    cells: list[EvalCell] = []
    for benchmark in config["benchmarks"]:
        # disabled 表示不进入默认全矩阵；显式 --benchmark 仍可用于配置和单条调试。
        if not benchmark.get("enabled", True) and benchmark_filter != benchmark["id"]:
            continue
        if benchmark_filter and benchmark["id"] != benchmark_filter:
            continue
        actor_model_ids = (
            [model for model in benchmark["actor_models"] if model in models]
            if benchmark["uses_llm"]
            else ["deterministic"]
        )
        task_ids = [task for task in benchmark.get("tasks", ["all"]) if not task_filter or task == task_filter]
        ablation_ids = [
            item
            for item in benchmark.get("ablations", ["full"])
            if not ablation_filter or item == ablation_filter
        ]
        for method_id in benchmark["methods"]:
            if method_id not in methods or (method_filter and method_id != method_filter):
                continue
            method_uses_sop_agent = methods[method_id].get("uses_sop_agent", False)
            if not benchmark["uses_llm"]:
                sop_model_ids = ["deterministic"]
            elif method_uses_sop_agent:
                sop_model_ids = [
                    model for model in benchmark.get("sop_models", []) if model in models
                ]
            else:
                # No-memory、G-Memory 等方法没有本项目的 SOP-Agent。使用显式占位符，
                # 避免把这些基线无意义地乘上全部 SOP 模型并重复计分。
                sop_model_ids = ["not-applicable"]
            for framework_id in benchmark["mas_frameworks"]:
                if framework_id not in frameworks or (mas_filter and framework_id != mas_filter):
                    continue
                allowed_frameworks = benchmark.get("method_mas", {}).get(method_id)
                if allowed_frameworks is not None and framework_id not in allowed_frameworks:
                    continue
                for task_id in task_ids:
                    if not _supported_by_benchmark_cell_rules(
                        benchmark, task_id, method_id, framework_id
                    ):
                        continue
                    for actor_model_id in actor_model_ids:
                        if actor_model_filter and actor_model_id != actor_model_filter:
                            continue
                        for seed in benchmark.get("seeds", [0]):
                            if seed_filter is not None and int(seed) != seed_filter:
                                continue
                            for sop_model_id in sop_model_ids:
                                # --sop-model 只选择 Team Memory 的 curator；没有 SOP-Agent
                                # 的基线仍需保留，才能在同一命令中形成完整配对主表。
                                if (
                                    method_uses_sop_agent
                                    and sop_model_filter
                                    and sop_model_id != sop_model_filter
                                ):
                                    continue
                                case_ids = _case_ids_for_task(benchmark, task_id, case_id)
                                for selected_case_id in case_ids:
                                    # ablation 放在最内层，使 --limit 优先得到同一 case
                                    # 的四个贡献条件，而不是不配对的模型或 run 配置。
                                    for ablation_id in ablation_ids:
                                        cells.append(
                                            EvalCell(
                                                benchmark=benchmark["id"],
                                                task=task_id,
                                                memory_method=method_id,
                                                actor_model=actor_model_id,
                                                sop_model=sop_model_id,
                                                mas_framework=framework_id,
                                                seed=int(seed),
                                                ablation=ablation_id,
                                                case_id=selected_case_id,
                                            )
                                        )
    return cells


class EvaluationState:
    """SQLite checkpoint；进程异常退出后 running 单元会在下次启动时重试。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA busy_timeout=30000")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluation_runs (
                cell_key TEXT PRIMARY KEY,
                cell_json TEXT NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                started_at REAL,
                finished_at REAL,
                result_path TEXT,
                log_path TEXT,
                error TEXT
            )
            """
        )
        self.connection.commit()

    def status(self, cell_key: str) -> str | None:
        row = self.connection.execute(
            "SELECT status FROM evaluation_runs WHERE cell_key = ?", (cell_key,)
        ).fetchone()
        return row[0] if row else None

    def start(self, cell: EvalCell, result_path: Path, log_path: Path) -> None:
        self.connection.execute(
            """
            INSERT INTO evaluation_runs(
                cell_key, cell_json, status, attempts, started_at, result_path, log_path
            ) VALUES (?, ?, 'running', 1, ?, ?, ?)
            ON CONFLICT(cell_key) DO UPDATE SET
                status='running', attempts=attempts+1, started_at=excluded.started_at,
                finished_at=NULL, result_path=excluded.result_path,
                log_path=excluded.log_path, error=NULL
            """,
            (
                cell.key,
                json.dumps(cell.__dict__, sort_keys=True),
                time.time(),
                str(result_path),
                str(log_path),
            ),
        )
        self.connection.commit()

    def finish(self, cell_key: str, status: str, error: str | None = None) -> None:
        self.connection.execute(
            "UPDATE evaluation_runs SET status=?, finished_at=?, error=? WHERE cell_key=?",
            (status, time.time(), error, cell_key),
        )
        self.connection.commit()

    def import_success(self, cell: EvalCell, result_path: Path, log_path: Path) -> None:
        """Mark a case-level cell as completed from a verified legacy result file."""
        self.connection.execute(
            """
            INSERT INTO evaluation_runs(
                cell_key, cell_json, status, attempts, started_at, finished_at,
                result_path, log_path, error
            ) VALUES (?, ?, 'success', 0, ?, ?, ?, ?, NULL)
            ON CONFLICT(cell_key) DO UPDATE SET
                cell_json=excluded.cell_json,
                status='success',
                finished_at=excluded.finished_at,
                result_path=excluded.result_path,
                log_path=excluded.log_path,
                error=NULL
            """,
            (
                cell.key,
                json.dumps(cell.__dict__, sort_keys=True),
                time.time(),
                time.time(),
                str(result_path),
                str(log_path),
            ),
        )
        self.connection.commit()

    def summary(self) -> dict[str, int]:
        rows = self.connection.execute(
            "SELECT status, COUNT(*) FROM evaluation_runs GROUP BY status"
        ).fetchall()
        return {status: count for status, count in rows}

    def rows(self) -> list[dict[str, Any]]:
        columns = [
            "cell_key",
            "cell_json",
            "status",
            "attempts",
            "started_at",
            "finished_at",
            "result_path",
            "log_path",
            "error",
        ]
        records = self.connection.execute(
            "SELECT cell_key, cell_json, status, attempts, started_at, finished_at, "
            "result_path, log_path, error FROM evaluation_runs ORDER BY cell_key"
        ).fetchall()
        return [dict(zip(columns, record)) for record in records]

    def close(self) -> None:
        self.connection.close()


def _lookup(config: dict[str, Any], table: str, row_id: str) -> dict[str, Any]:
    return next(row for row in config[table] if row["id"] == row_id)


def _model_matches(config: dict[str, Any], expected_id: str, observed: Any) -> bool:
    """Match either the matrix model id or its provider-facing API model name."""
    if observed is None:
        return False
    observed_text = str(observed)
    if observed_text == expected_id:
        return True
    for model in config.get("models", []):
        if model.get("id") == expected_id:
            return observed_text == model.get("api_model")
    return False


def _context_matches_cell(
    config: dict[str, Any],
    benchmark: dict[str, Any],
    cell: EvalCell,
    context: dict[str, Any],
) -> bool:
    aliases = {cell.benchmark, *benchmark.get("legacy_benchmark_ids", [])}
    return (
        context.get("benchmark") in aliases
        and context.get("task") == cell.task
        and context.get("memory_method") == cell.memory_method
        and _model_matches(config, cell.actor_model, context.get("actor_model"))
        and _model_matches(config, cell.sop_model, context.get("sop_model"))
        and context.get("mas_framework") == cell.mas_framework
        and str(context.get("seed")) == str(cell.seed)
        and context.get("ablation", "full") == cell.ablation
    )


def _context_case_matches(context: dict[str, Any], cell: EvalCell) -> bool:
    observed = context.get("case_id")
    if cell.case_id:
        return observed is not None and str(observed) == cell.case_id
    return observed in (None, "")


def _existing_result_is_usable(
    config: dict[str, Any], cell: EvalCell, result_path: Path
) -> tuple[bool, str]:
    if not result_path.exists():
        return False, "missing"
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return False, f"invalid JSON: {error}"
    context = _context_from_result(payload)
    if context is None:
        return False, "missing context"
    benchmark = _lookup(config, "benchmarks", cell.benchmark)
    if not _context_matches_cell(config, benchmark, cell, context):
        return False, "context mismatch"
    if not _context_case_matches(context, cell):
        return False, "case mismatch"
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return False, "missing metrics"
    if "primary_score" not in metrics:
        return False, "missing metrics.primary_score"
    if float(metrics.get("case_count", 0.0) or 0.0) != 1.0:
        return False, "metrics.case_count is not 1.0"
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        return False, "missing cases"
    if cell.task == "webarena":
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            return False, "missing metadata"
        if metadata.get("execution_mode") != "official-webarena-single-case":
            return False, "not official WebArena runtime output"
        if not metadata.get("official_entrypoint") or not metadata.get("official_log"):
            return False, "missing official WebArena provenance"
    return True, "ok"


def _model_connection(
    config: dict[str, Any], model: dict[str, Any]
) -> dict[str, str | bool]:
    """解析模型自己的 OpenAI-compatible endpoint，未声明时继承全局中转站。

    本地 vLLM 通常不校验 key，因此可把 ``api_key_required`` 设为 false；OpenAI SDK
    仍要求传入一个非空字符串，此时只在进程内使用固定占位值，不写入结果或日志。
    """
    provider = config["provider"]
    base_url_env = model.get("base_url_env", provider["base_url_env"])
    key_env = model.get("key_env", provider["key_env"])
    base_url = os.environ.get(base_url_env, model.get("base_url_default"))
    if not base_url and base_url_env == provider["base_url_env"]:
        base_url = provider.get("base_url_default")
    if not base_url or not base_url.rstrip("/").endswith("/v1"):
        raise ValueError(
            f"model {model['id']} OpenAI-compatible base URL must end with /v1; "
            f"set {base_url_env}"
        )
    key_required = model.get("api_key_required", True)
    api_key = os.environ.get(key_env)
    if key_required and not api_key:
        raise RuntimeError(f"model {model['id']} missing environment variable: {key_env}")
    return {
        "base_url": base_url,
        "base_url_env": base_url_env,
        "api_key": api_key or "local-no-key",
        "key_env": key_env,
        "key_required": key_required,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def smoke_llm(config: dict[str, Any], model_id: str, prompt: str) -> dict[str, Any]:
    """发起一次最小 Chat Completions 调用，验证 relay、key 和模型名。"""
    try:
        from openai import OpenAI
    except ImportError as error:
        message = "install evaluation dependencies with: pip install -e '.[eval]'"
        raise RuntimeError(message) from error
    model = _lookup(config, "models", model_id)
    provider = config["provider"]
    connection = _model_connection(config, model)
    client = OpenAI(
        api_key=str(connection["api_key"]),
        base_url=str(connection["base_url"]),
        timeout=provider.get("timeout_seconds", 120),
        max_retries=provider.get("max_retries", 2),
    )
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model["api_model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=model.get("temperature", 0.0),
        max_tokens=model.get("max_tokens", 1024),
    )
    usage = getattr(response, "usage", None)
    return {
        "ok": True,
        "provider": provider["id"],
        "base_url": connection["base_url"],
        "model_id": model_id,
        "api_model": model["api_model"],
        "latency_seconds": time.perf_counter() - started,
        "text": response.choices[0].message.content,
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        },
    }


def _matrix_root(config: dict[str, Any]) -> Path:
    """返回配置文件所在目录；仓库、日志和结果路径都以它为基准。"""
    return Path(config["_source"]).parent


def _resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _matrix_root(config) / path


def _state_db_path(config: dict[str, Any]) -> Path:
    override = os.environ.get("TEAM_MEMORY_EVAL_STATE_DB")
    return _resolve_path(config, override or config["state_db"])


def _render_command(
    parts: list[str],
    cell: EvalCell,
    output_path: Path,
    *,
    repository: Path,
    project_root: Path,
    actor_api_model: str,
    sop_api_model: str,
) -> list[str]:
    values = {
        **cell.__dict__,
        "case_id": cell.case_id or "",
        "output": str(output_path.resolve()),
        "repository": str(repository),
        "project_root": str(project_root),
        # api_model 保留给 G-Memory 等只接受一个 --model 的官方脚本；它始终表示
        # 执行 benchmark 的 Actor，而不是 Team Memory 的 SOP-Agent。
        "api_model": actor_api_model,
        "actor_api_model": actor_api_model,
        "sop_api_model": sop_api_model,
    }
    return [part.format(**values) for part in parts]


def _method_job(benchmark: dict[str, Any], method: str) -> dict[str, Any] | None:
    """选择 benchmark 的方法专用作业，避免不同方法误跑同一个官方脚本。"""
    job = benchmark.get("method_jobs", {}).get(method)
    if job is not None and not isinstance(job, dict):
        raise ValueError(f"{benchmark['id']}.method_jobs.{method} must be an object")
    return job


def _command_steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    if "steps" in job:
        steps = job["steps"]
    elif "command" in job:
        steps = [{"command": job["command"]}]
    else:
        raise ValueError("external method job requires command or steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("external method job steps must be a non-empty list")
    for step in steps:
        if not isinstance(step, dict) or not isinstance(step.get("command"), list):
            raise ValueError("each external step requires a command argument array")
        if not step["command"] or not all(isinstance(part, str) for part in step["command"]):
            raise ValueError("external command arguments must be non-empty strings")
    return steps


def _prepend_conda(command: list[str], environment: str | None) -> list[str]:
    """通过 conda run 调用隔离环境；不依赖交互式 conda activate。"""
    if not environment:
        return command
    return ["conda", "run", "--no-capture-output", "-n", environment, *command]


def run_cell(
    config: dict[str, Any], cell: EvalCell, result_path: Path, log_path: Path
) -> str:
    """运行一个矩阵单元；外部方法使用自己的命令、Conda 环境和工作目录。"""
    benchmark = _lookup(config, "benchmarks", cell.benchmark)
    if benchmark["runner"] == "internal":
        _write_json(
            result_path,
            {
                "cell": cell.__dict__,
                "result": run_all_benchmarks(include_latency=cell.case_id is None),
            },
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("internal benchmark completed\n", encoding="utf-8")
        return SUCCESS
    job = _method_job(benchmark, cell.memory_method)
    if job is None:
        _write_json(
            result_path,
            {
                "cell": cell.__dict__,
                "status": "not_configured",
                "reason": "no method-specific evaluation job is defined",
                "repository": benchmark.get("repository"),
            },
        )
        return "not_configured"
    env = os.environ.copy()
    provider = config["provider"]
    actor_model = _lookup(config, "models", cell.actor_model)
    sop_model = (
        _lookup(config, "models", cell.sop_model)
        if cell.sop_model not in {"not-applicable", "deterministic"}
        else None
    )
    actor_connection = _model_connection(config, actor_model)
    sop_connection = (
        _model_connection(config, sop_model) if sop_model is not None else actor_connection
    )
    project_root = _matrix_root(config)
    repository = _resolve_path(config, benchmark["repository"])
    if not repository.is_dir():
        raise FileNotFoundError(f"benchmark repository not found: {repository}")
    env.update(
        {
            "OPENAI_BASE_URL": os.environ.get(
                str(actor_connection["base_url_env"]), str(actor_connection["base_url"])
            ),
            # 旧 benchmark 只读取标准 OpenAI 变量，因此映射为 Actor 连接；Team Memory
            # adapter 使用下面分开的 Actor/SOP 变量，避免两个角色串用 endpoint。
            "OPENAI_API_KEY": str(actor_connection["api_key"]),
            "TEAM_MEMORY_EVAL_ACTOR_BASE_URL": str(actor_connection["base_url"]),
            "TEAM_MEMORY_EVAL_ACTOR_API_KEY": str(actor_connection["api_key"]),
            "TEAM_MEMORY_EVAL_ACTOR_KEY_ENV": str(actor_connection["key_env"]),
            "TEAM_MEMORY_EVAL_SOP_BASE_URL": str(sop_connection["base_url"]),
            "TEAM_MEMORY_EVAL_SOP_API_KEY": str(sop_connection["api_key"]),
            "TEAM_MEMORY_EVAL_SOP_KEY_ENV": str(sop_connection["key_env"]),
            "TEAM_MEMORY_EVAL_ACTOR_MODEL": actor_model["api_model"],
            "TEAM_MEMORY_EVAL_SOP_MODEL": (
                sop_model["api_model"] if sop_model is not None else cell.sop_model
            ),
            # 第三方官方脚本通常只有一个模型参数；兼容变量明确指向 Actor。
            "TEAM_MEMORY_EVAL_MODEL": actor_model["api_model"],
            "TEAM_MEMORY_EVAL_METHOD": cell.memory_method,
            "TEAM_MEMORY_EVAL_MAS": cell.mas_framework,
            # 保留旧变量只为兼容尚未迁移的第三方脚本；新适配器应读取 EVAL_MAS。
            "TEAM_MEMORY_EVAL_AGENT": cell.mas_framework,
            "TEAM_MEMORY_EVAL_BENCHMARK": cell.benchmark,
            "TEAM_MEMORY_EVAL_TASK": cell.task,
            "TEAM_MEMORY_EVAL_ABLATION": cell.ablation,
            "TEAM_MEMORY_EVAL_SEED": str(cell.seed),
            "TEAM_MEMORY_EVAL_CASE_ID": cell.case_id or "",
            "TEAM_MEMORY_EVAL_OUTPUT": str(result_path.resolve()),
        }
    )
    if cell.sop_model in {
        "qwen3.5-9b",
        "qwen3.5-27b",
        "gemma-4-12b-it",
        "gemma-4-31b-it",
    }:
        env["TEAM_MEMORY_EVAL_SOP_RESPONSE_FORMAT"] = "json_object"
    # 外部命令的 cwd 是第三方仓库。把共享数据库转成相对主项目根目录的绝对路径，
    # 否则同一个 TEAM_MEMORY_DB 会在 GMemory/MARBLE 目录各创建一份，破坏共享记忆实验。
    database_value = os.environ.get("TEAM_MEMORY_DB")
    if database_value:
        env["TEAM_MEMORY_DB"] = str(_resolve_path(config, database_value).resolve())
    # G-Memory 等旧版项目使用 OPENAI_API_BASE；两者指向同一 relay。
    env["OPENAI_API_BASE"] = env["OPENAI_BASE_URL"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        for index, step in enumerate(_command_steps(job), 1):
            workdir_value = step.get("workdir", job.get("workdir", "."))
            workdir = Path(workdir_value)
            if not workdir.is_absolute():
                workdir = repository / workdir
            if not workdir.is_dir():
                raise FileNotFoundError(f"external step workdir not found: {workdir}")
            command = _render_command(
                step["command"],
                cell,
                result_path,
                repository=repository,
                project_root=project_root,
                actor_api_model=actor_model["api_model"],
                sop_api_model=(
                    sop_model["api_model"] if sop_model is not None else cell.sop_model
                ),
            )
            command = _prepend_conda(
                command, step.get("conda_env", job.get("conda_env"))
            )
            log.write(f"[step {index}] cwd={workdir}\n")
            log.write(f"[step {index}] command={json.dumps(command, ensure_ascii=False)}\n")
            log.flush()
            completed = subprocess.run(
                command,
                cwd=workdir,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=step.get("timeout_seconds", job.get("timeout_seconds")),
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"external step {index} exited with code {completed.returncode}; "
                    f"see {log_path}"
                )
    if not result_path.exists():
        raise RuntimeError(
            "external job completed but did not write TEAM_MEMORY_EVAL_OUTPUT; "
            "add a final normalization step"
        )
    return SUCCESS


def run_matrix(
    config: dict[str, Any],
    cells: list[EvalCell],
    *,
    resume: bool = True,
    rerun_failed: bool = False,
    skip_existing_results: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    output_dir = _resolve_path(config, config["output_dir"])
    state = EvaluationState(_state_db_path(config))
    try:
        for cell in cells:
            previous = state.status(cell.key)
            if resume and previous == SUCCESS:
                continue
            if resume and previous == "failed" and not rerun_failed:
                continue
            result_path = output_dir / "results" / f"{cell.key}.json"
            log_path = output_dir / "logs" / f"{cell.key}.log"
            if resume and skip_existing_results:
                usable, reason = _existing_result_is_usable(config, cell, result_path)
                if usable:
                    if not dry_run:
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        if not log_path.exists():
                            log_path.write_text(
                                f"skipped existing result {result_path}\n",
                                encoding="utf-8",
                            )
                        state.import_success(cell, result_path, log_path)
                    print(f"SKIPPED existing {cell.key}")
                    continue
                if result_path.exists():
                    raise RuntimeError(
                        f"existing result for {cell.key} is not usable for skip: {reason}"
                    )
            if dry_run:
                print(json.dumps({"cell_key": cell.key, **cell.__dict__}, ensure_ascii=False))
                continue
            state.start(cell, result_path, log_path)
            try:
                status = run_cell(config, cell, result_path, log_path)
            except Exception as error:
                state.finish(cell.key, "failed", str(error))
                print(f"FAILED {cell.key}: {error}", file=sys.stderr)
            else:
                state.finish(cell.key, status)
                print(f"{status.upper()} {cell.key}")
        return state.summary()
    finally:
        state.close()


def _context_from_result(payload: dict[str, Any]) -> dict[str, Any] | None:
    context = payload.get("context")
    if isinstance(context, dict):
        return context
    cell = payload.get("cell")
    return cell if isinstance(cell, dict) else None


def _case_matches(case: dict[str, Any], case_id: str) -> bool:
    identifiers = {
        str(case.get("case_id", "")),
        str(case.get("official_task_index", "")),
        str(case.get("id", "")),
    }
    return case_id in identifiers


def _case_metrics(benchmark: dict[str, Any], task: str, case: dict[str, Any]) -> dict[str, float]:
    reward = float(case.get("reward", 0.0) or 0.0)
    done = 1.0 if case.get("done") else 0.0
    task_metric = benchmark.get("task_metrics", {}).get(task, "primary_score")
    primary = reward if task_metric == "progress_rate" else done
    return {
        task_metric: primary,
        "primary_score": primary,
        "case_count": 1.0,
        "mean_reward": reward,
        "success_rate": done,
    }


def import_legacy_results(
    config: dict[str, Any],
    cells: list[EvalCell],
    *,
    legacy_results_dir: str | Path,
    dry_run: bool = False,
) -> dict[str, int]:
    """Import verified legacy results into the current checkpoint state.

    Exact full-run checkpoints are preserved as full-run cells. Episode-level
    payloads with per-case records can also be split into case-level checkpoints.
    Model matching accepts both matrix ids and provider API model names so older
    result files such as ``Qwen/Qwen3.5-27B`` still match ``qwen3.5-27b``.
    """
    legacy_dir = _resolve_path(config, legacy_results_dir)
    if not legacy_dir.is_dir():
        raise FileNotFoundError(f"legacy results directory not found: {legacy_dir}")
    payloads: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for path in sorted(legacy_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        context = _context_from_result(payload)
        if context is not None:
            payloads.append((path, payload, context))
    output_dir = _resolve_path(config, config["output_dir"])
    state = EvaluationState(_state_db_path(config))
    counts = {"imported": 0, "skipped": 0, "missing": 0}
    try:
        for cell in cells:
            if state.status(cell.key) == SUCCESS:
                counts["skipped"] += 1
                continue
            benchmark = _lookup(config, "benchmarks", cell.benchmark)
            exact_match: tuple[Path, dict[str, Any], dict[str, Any]] | None = None
            case_match: tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]] | None = None
            for path, payload, context in payloads:
                if not _context_matches_cell(config, benchmark, cell, context):
                    continue
                metric_source = payload.get("metrics", payload.get("result"))
                if isinstance(metric_source, dict) and _context_case_matches(context, cell):
                    exact_match = (path, payload, context)
                    break
                if cell.case_id:
                    for case in payload.get("cases", []):
                        if isinstance(case, dict) and _case_matches(case, cell.case_id):
                            case_match = (path, payload, context, case)
                            break
                    if case_match:
                        break
            if exact_match is None and case_match is None:
                counts["missing"] += 1
                continue
            result_path = output_dir / "results" / f"{cell.key}.json"
            log_path = output_dir / "logs" / f"{cell.key}.log"
            if exact_match is not None:
                source_path, payload, _context = exact_match
                imported = {
                    "context": {**cell.__dict__, "output_path": str(result_path)},
                    "metrics": payload.get("metrics", payload.get("result", {})),
                    "cases": payload.get("cases", []),
                    "metadata": {
                        **payload.get("metadata", {}),
                        "imported_from": str(source_path),
                        "import_policy": "legacy full result imported as matching checkpoint",
                    },
                }
            else:
                source_path, _payload, _context, case = case_match
                imported = {
                    "context": {**cell.__dict__, "output_path": str(result_path)},
                    "metrics": _case_metrics(benchmark, cell.task, case),
                    "cases": [case],
                    "metadata": {
                        "imported_from": str(source_path),
                        "import_policy": "legacy episode result split into case-level checkpoint",
                    },
                }
            if dry_run:
                print(json.dumps({"would_import": cell.key, "source": str(source_path)}, ensure_ascii=False))
            else:
                if source_path.resolve() != result_path.resolve():
                    _write_json(result_path, imported)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(f"imported from {source_path}\n", encoding="utf-8")
                state.import_success(cell, result_path, log_path)
            counts["imported"] += 1
        return counts
    finally:
        state.close()


def doctor(config: dict[str, Any]) -> dict[str, Any]:
    provider = config["provider"]
    return {
        "matrix": config["_source"],
        "provider": provider["id"],
        "base_url": os.environ.get(
            provider["base_url_env"], provider.get("base_url_default")
        ),
        "key_env": provider["key_env"],
        "key_present": bool(os.environ.get(provider["key_env"])),
        "conda_executable": shutil.which("conda"),
        "models": [
            {
                "id": model["id"],
                "api_model": model["api_model"],
                "enabled": model.get("enabled", True),
                "placeholder": model["api_model"].startswith("REPLACE_"),
                "base_url_env": model.get("base_url_env", provider["base_url_env"]),
                "base_url": os.environ.get(
                    model.get("base_url_env", provider["base_url_env"]),
                    model.get("base_url_default", provider.get("base_url_default")),
                ),
                "key_env": model.get("key_env", provider["key_env"]),
            }
            for model in config["models"]
        ],
        "benchmarks": [
            {
                "id": benchmark["id"],
                "runner": benchmark["runner"],
                "configured_methods": sorted(benchmark.get("method_jobs", {})),
                "matrix_methods": benchmark["methods"],
                "conda_environments": sorted(
                    {
                        job["conda_env"]
                        for job in benchmark.get("method_jobs", {}).values()
                        if job.get("conda_env")
                    }
                ),
                "repository_present": (
                    _resolve_path(config, benchmark["repository"]).is_dir()
                    if benchmark.get("repository")
                    else True
                ),
            }
            for benchmark in config["benchmarks"]
        ],
    }


def source_reproduction_commands(
    config: dict[str, Any], benchmark_filter: str | None = None
) -> list[dict[str, Any]]:
    """返回官方仓库复现入口；这些命令用于核对原始实现，不展开论文方法矩阵。"""
    workflows = config.get("source_reproductions", [])
    if not isinstance(workflows, list):
        raise ValueError("source_reproductions must be a list")
    selected = []
    for workflow in workflows:
        if benchmark_filter and workflow["id"] != benchmark_filter:
            continue
        repository = _resolve_path(config, workflow["repository"])
        selected.append(
            {
                **workflow,
                "repository": str(repository),
                "repository_present": repository.is_dir(),
            }
        )
    return selected


def _numeric_metrics(payload: Any, prefix: str = "") -> dict[str, float]:
    """递归提取数值指标；布尔值不作为 0/1 混入论文汇总。"""
    metrics: dict[str, float] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            child = f"{prefix}.{key}" if prefix else key
            metrics.update(_numeric_metrics(value, child))
    elif isinstance(payload, (int, float)) and not isinstance(payload, bool):
        metrics[prefix] = float(payload)
    return metrics


def _mean_ci95(values: list[float], *, iterations: int = 2_000) -> dict[str, float]:
    """对完整 run/case 指标做确定性 bootstrap，返回均值、95% CI 和样本数。

    这里的输入是一组完整 run 的指标，而不是把同一 run 内的 token/step 当成独立样本。
    当矩阵只使用固定 seed=0 时，CI 只能反映 matched case 层面的波动，不能解释为
    随机 seed 稳定性。
    """
    if not values:
        raise ValueError("cannot summarize an empty metric")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean, "n": 1.0}
    # 固定随机种子使报告可复现；采样次数只影响 CI 的数值精度，不影响原始 run。
    generator = random.Random(0)
    bootstrap = sorted(
        sum(generator.choice(values) for _ in values) / len(values)
        for _ in range(iterations)
    )
    low_index = int(0.025 * (iterations - 1))
    high_index = int(0.975 * (iterations - 1))
    return {
        "mean": mean,
        "ci95_low": bootstrap[low_index],
        "ci95_high": bootstrap[high_index],
        "n": float(len(values)),
    }


def build_report(config: dict[str, Any]) -> dict[str, Any]:
    """读取 checkpoint/result，输出逐 run CSV 和按实验条件聚合的 JSON。"""
    state = EvaluationState(_state_db_path(config))
    try:
        state_rows = state.rows()
    finally:
        state.close()
    detailed: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str, str, str, str, str], list[dict[str, float]]] = {}
    for row in state_rows:
        cell = json.loads(row["cell_json"])
        metrics: dict[str, float] = {}
        result_path = Path(row["result_path"]) if row["result_path"] else None
        if row["status"] == SUCCESS and result_path and result_path.exists():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            # 外部适配器应把论文指标放在 metrics；内部报告保留完整数值路径。
            source = payload.get("metrics", payload.get("result", {}))
            metrics = _numeric_metrics(source)
            group = (
                cell["benchmark"],
                cell["task"],
                cell["memory_method"],
                cell["actor_model"],
                cell["sop_model"],
                cell["mas_framework"],
                cell["ablation"],
            )
            grouped.setdefault(group, []).append(metrics)
        detailed.append(
            {
                **cell,
                "cell_key": row["cell_key"],
                "status": row["status"],
                "attempts": row["attempts"],
                "result_path": row["result_path"],
                "error": row["error"],
                "metrics": metrics,
            }
        )
    summaries: list[dict[str, Any]] = []
    for group, metric_rows in sorted(grouped.items()):
        names = sorted({name for metrics in metric_rows for name in metrics})
        means = {
            name: sum(row[name] for row in metric_rows if name in row)
            / sum(name in row for row in metric_rows)
            for name in names
        }
        metric_stats = {
            name: _mean_ci95([row[name] for row in metric_rows if name in row])
            for name in names
        }
        summaries.append(
            {
                "benchmark": group[0],
                "task": group[1],
                "memory_method": group[2],
                "actor_model": group[3],
                "sop_model": group[4],
                "mas_framework": group[5],
                "ablation": group[6],
                "successful_runs": len(metric_rows),
                "mean_metrics": means,
                "metric_stats": metric_stats,
            }
        )
    output_dir = _resolve_path(config, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "runs.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        fields = [
            "cell_key",
            "benchmark",
            "task",
            "memory_method",
            "actor_model",
            "sop_model",
            "mas_framework",
            "ablation",
            "seed",
            "case_id",
            "status",
            "attempts",
            "result_path",
            "error",
            "metrics_json",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in detailed:
            writer.writerow(
                {
                    **{field: row.get(field) for field in fields if field != "metrics_json"},
                    "metrics_json": json.dumps(row["metrics"], sort_keys=True),
                }
            )
    report = {
        "status_counts": {
            status: sum(row["status"] == status for row in detailed)
            for status in sorted({row["status"] for row in detailed})
        },
        "groups": summaries,
        "runs_csv": str(csv_path),
    }
    _write_json(output_dir / "summary.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified Team Memory evaluation matrix runner")
    parser.add_argument(
        "action",
        choices=(
            "doctor",
            "commands",
            "matrix",
            "smoke",
            "run",
            "status",
            "report",
            "import-legacy",
        ),
    )
    parser.add_argument("--config", type=Path, default=Path("evaluation_matrix.json"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--benchmark")
    parser.add_argument("--task")
    parser.add_argument("--memory-method")
    parser.add_argument("--actor-model")
    parser.add_argument("--sop-model")
    parser.add_argument("--model", help="model id used only by the smoke action")
    parser.add_argument("--mas", dest="mas_framework")
    parser.add_argument("--ablation")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--case-id")
    parser.add_argument(
        "--legacy-results-dir",
        type=Path,
        default=Path("benchmark-results/unified-v3/results"),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument(
        "--skip-existing-results",
        action="store_true",
        help="skip an already-written verified result JSON even when using a fresh state DB",
    )
    parser.add_argument(
        "--state-db",
        type=Path,
        help="override checkpoint SQLite path for isolated smoke/resume runs",
    )
    parser.add_argument("--prompt", default="只回复 OK")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_env_file(args.env_file)
    config = load_matrix(args.config)
    if args.state_db is not None:
        os.environ["TEAM_MEMORY_EVAL_STATE_DB"] = str(args.state_db)
    if args.action == "doctor":
        print(json.dumps(doctor(config), ensure_ascii=False, indent=2))
        return 0
    if args.action == "commands":
        commands = source_reproduction_commands(config, args.benchmark)
        print(json.dumps(commands, ensure_ascii=False, indent=2))
        return 0
    if args.action == "smoke":
        if not args.model:
            raise SystemExit("smoke requires --model")
        result = smoke_llm(config, args.model, args.prompt)
        output = _resolve_path(config, config["output_dir"]) / "smoke" / f"{args.model}.json"
        _write_json(output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.action == "status":
        state = EvaluationState(_state_db_path(config))
        try:
            print(json.dumps(state.summary(), ensure_ascii=False, indent=2))
        finally:
            state.close()
        return 0
    if args.action == "report":
        print(json.dumps(build_report(config), ensure_ascii=False, indent=2))
        return 0
    cells = expand_matrix(
        config,
        benchmark_filter=args.benchmark,
        task_filter=args.task,
        method_filter=args.memory_method,
        actor_model_filter=args.actor_model,
        sop_model_filter=args.sop_model,
        mas_filter=args.mas_framework,
        ablation_filter=args.ablation,
        seed_filter=args.seed,
        case_id=args.case_id,
    )
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be >= 1")
        cells = cells[: args.limit]
    if args.action == "matrix":
        for cell in cells:
            print(json.dumps({"cell_key": cell.key, **cell.__dict__}, ensure_ascii=False))
        print(f"cells={len(cells)}", file=sys.stderr)
        return 0
    if args.action == "import-legacy":
        summary = import_legacy_results(
            config,
            cells,
            legacy_results_dir=args.legacy_results_dir,
            dry_run=args.dry_run,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    summary = run_matrix(
        config,
        cells,
        resume=not args.no_resume,
        rerun_failed=args.rerun_failed,
        skip_existing_results=args.skip_existing_results,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    # 让 nohup/Slurm/bash 能可靠判断整批任务是否真的成功；以前即使所有外部
    # adapter 都是 not_configured，进程仍返回 0，容易产生“实验已经跑完”的假象。
    return 1 if summary.get("failed", 0) or summary.get("not_configured", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
