"""把第三方 benchmark 的 JSON/JSONL 结果转换成统一评测结果。

该模块作为外部 ``method_jobs.steps`` 的最后一步运行。指标选择器使用点分路径；路径经过
列表时会展开全部元素，因此既能读取单个汇总 JSON，也能对逐 case JSONL 求均值。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from .evaluation_adapter import EvaluationContext


def _select(value: Any, path: str) -> list[Any]:
    """沿点分路径选择值，并自动展开中途出现的列表。"""
    parts = [part for part in path.split(".") if part]
    current = [value]
    for part in parts:
        following: list[Any] = []
        for item in current:
            if isinstance(item, list):
                for child in item:
                    if isinstance(child, dict) and part in child:
                        following.append(child[part])
            elif isinstance(item, dict) and part in item:
                following.append(item[part])
        current = following
    flattened: list[Any] = []
    for item in current:
        flattened.extend(item if isinstance(item, list) else [item])
    return flattened


def load_native_result(path: Path, result_format: str) -> tuple[Any, list[dict[str, Any]]]:
    """读取原生 JSON 或 JSONL，并返回用于选指标的对象及逐 case 记录。"""
    if result_format == "json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        cases = payload if isinstance(payload, list) else payload.get("cases", [])
        return payload, [row for row in cases if isinstance(row, dict)]
    if result_format == "jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return rows, [row for row in rows if isinstance(row, dict)]
    raise ValueError(f"unsupported result format: {result_format}")


def extract_metrics(payload: Any, specifications: list[str]) -> dict[str, float]:
    """解析 ``name=path`` 或 ``name=path:sum`` 指标定义。"""
    metrics: dict[str, float] = {}
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(f"metric must be NAME=PATH[:mean|sum|first]: {specification}")
        name, selector = specification.split("=", 1)
        path, separator, aggregation = selector.rpartition(":")
        if not separator or aggregation not in {"mean", "sum", "first"}:
            path, aggregation = selector, "mean"
        values = [
            float(value)
            for value in _select(payload, path)
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if not values:
            raise ValueError(f"metric selector returned no numeric values: {selector}")
        if aggregation == "sum":
            metrics[name] = sum(values)
        elif aggregation == "first":
            metrics[name] = values[0]
        else:
            metrics[name] = fmean(values)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize an external benchmark result")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--format", choices=("json", "jsonl"), default="json")
    parser.add_argument("--metric", action="append", required=True)
    parser.add_argument("--case-id-field")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload, cases = load_native_result(args.input, args.format)
    if args.case_id_field:
        cases = [case for case in cases if args.case_id_field in case]
    context = EvaluationContext.from_environment()
    context.write_result(
        extract_metrics(payload, args.metric),
        cases=cases,
        metadata={
            "native_result": str(args.input.resolve()),
            "native_format": args.format,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
