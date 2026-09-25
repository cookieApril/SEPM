"""在消耗 API 额度前验证论文实验服务器是否真正可运行。

该脚本只做只读检查：环境变量、模型占位符、第三方仓库路径以及 method_jobs。外部适配器
尚未接好时返回非零退出码，避免 runner 把大量单元记成 ``not_configured``。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from team_memory.evaluation_runner import _model_connection, load_env_file, load_matrix


def _resolve(config: dict[str, Any], value: str) -> Path:
    """所有相对路径均以 evaluation_matrix.json 所在目录为准。"""
    source = Path(config["_source"]).parent
    path = Path(value)
    return path if path.is_absolute() else source / path


def _placeholder_values(value: Any) -> list[str]:
    """递归找出会让外部 job 必然失败的模板值。"""
    if isinstance(value, dict):
        return [item for nested in value.values() for item in _placeholder_values(nested)]
    if isinstance(value, list):
        return [item for nested in value for item in _placeholder_values(nested)]
    if not isinstance(value, str):
        return []
    normalized = value.replace("\\", "/")
    return [value] if "path/to/" in normalized or value.startswith("REPLACE_") else []


def _case_id_errors(benchmark: dict[str, Any]) -> list[str]:
    """Paper experiments must checkpoint at benchmark-case granularity."""
    case_ids = benchmark.get("case_ids")
    tasks = benchmark.get("tasks", [])
    if not case_ids:
        return [f"{benchmark['id']}: missing case_ids; add matched case subset before running"]
    if isinstance(case_ids, list):
        return [] if case_ids else [f"{benchmark['id']}: case_ids list is empty"]
    if not isinstance(case_ids, dict):
        return [f"{benchmark['id']}: case_ids must be a list or task->list object"]
    errors: list[str] = []
    for task in tasks:
        values = case_ids.get(task)
        if not isinstance(values, list) or not values:
            errors.append(f"{benchmark['id']}: missing non-empty case_ids for task {task!r}")
    return errors


def inspect_setup(
    config: dict[str, Any],
    *,
    benchmark_ids: set[str],
    require_jobs: bool,
) -> dict[str, Any]:
    """返回可机器读取的检查报告；errors 非空时不得启动正式实验。"""
    errors: list[str] = []
    warnings: list[str] = []
    provider = config["provider"]
    key_name = provider["key_env"]
    base_name = provider["base_url_env"]
    base_url = os.environ.get(base_name, provider.get("base_url_default", ""))

    if not os.environ.get(key_name):
        errors.append(f"missing {key_name}; copy .env.example to .env and fill it on server")
    if not base_url.rstrip("/").endswith("/v1"):
        errors.append(f"{base_name} must end with /v1: {base_url!r}")

    model_report: list[dict[str, Any]] = []
    for model in config["models"]:
        if model.get("enabled", True) and model["api_model"].startswith("REPLACE_"):
            errors.append(f"model {model['id']} still uses a placeholder api_model")
        if not model.get("enabled", True):
            continue
        try:
            connection = _model_connection(config, model)
        except (RuntimeError, ValueError) as error:
            errors.append(str(error))
            continue
        model_report.append(
            {
                "id": model["id"],
                "api_model": model["api_model"],
                "family": model.get("family"),
                "base_url": connection["base_url"],
                "key_env": connection["key_env"],
                "key_present_or_optional": (
                    bool(os.environ.get(str(connection["key_env"])))
                    or not bool(connection["key_required"])
                ),
            }
        )

    selected = [
        benchmark
        for benchmark in config["benchmarks"]
        if benchmark["id"] in benchmark_ids
    ]
    missing_ids = benchmark_ids - {benchmark["id"] for benchmark in selected}
    if missing_ids:
        errors.append(f"unknown benchmark ids: {sorted(missing_ids)}")

    benchmark_report: list[dict[str, Any]] = []
    for benchmark in selected:
        repository_value = benchmark.get("repository")
        repository = _resolve(config, repository_value) if repository_value else None
        repository_present = repository is None or repository.is_dir()
        if not repository_present:
            errors.append(f"{benchmark['id']}: repository not found: {repository}")

        configured = set(benchmark.get("method_jobs", {}))
        required = set(benchmark.get("paper_run_methods", benchmark["methods"]))
        missing_jobs = sorted(required - configured)
        placeholder_jobs = {
            method: _placeholder_values(job)
            for method, job in benchmark.get("method_jobs", {}).items()
            if _placeholder_values(job)
        }
        if require_jobs and missing_jobs:
            errors.append(
                f"{benchmark['id']}: method_jobs missing for {missing_jobs}; "
                "implement/declare adapters before the full run"
            )
        elif missing_jobs:
            warnings.append(f"{benchmark['id']}: unconfigured methods {missing_jobs}")
        if placeholder_jobs:
            errors.append(
                f"{benchmark['id']}: placeholder values remain in method_jobs: "
                f"{placeholder_jobs}"
            )
        if require_jobs:
            errors.extend(_case_id_errors(benchmark))

        benchmark_report.append(
            {
                "id": benchmark["id"],
                "repository": str(repository) if repository else None,
                "repository_present": repository_present,
                "configured_methods": sorted(configured),
                "missing_method_jobs": missing_jobs,
                "placeholder_method_jobs": placeholder_jobs,
                "case_ids_configured": bool(benchmark.get("case_ids")),
            }
        )

    return {
        "ok": not errors,
        "provider": provider["id"],
        "base_url": base_url,
        "api_key_present": bool(os.environ.get(key_name)),
        "models": model_report,
        "benchmarks": benchmark_report,
        "warnings": warnings,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the paper experiment server")
    parser.add_argument("--config", type=Path, default=Path("evaluation_matrix.json"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--benchmark",
        action="append",
        dest="benchmarks",
        help="benchmark id; repeat the flag to check several",
    )
    parser.add_argument("--require-jobs", action="store_true")
    args = parser.parse_args()

    load_env_file(args.env_file)
    config = load_matrix(args.config)
    benchmark_ids = set(
        args.benchmarks
        or ["cross-benchmark-generality", "sop-model-sensitivity", "component-ablation"]
    )
    report = inspect_setup(
        config,
        benchmark_ids=benchmark_ids,
        require_jobs=args.require_jobs,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
