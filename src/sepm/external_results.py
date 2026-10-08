"""Convert third-party benchmark JSON/JSONL into unified evaluation results.

This module runs as the last external ``method_jobs.steps`` stage. Metric
selectors use dotted paths and expand list elements, supporting both aggregate
JSON files and means over per-case JSONL records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from .evaluation_adapter import EvaluationContext


def _select(value: Any, path: str) -> list[Any]:
    """Select values along a dotted path and expand intermediate lists."""
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
    """Read native JSON/JSONL and return the metric payload and per-case records."""
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
    """Parse ``name=path`` or ``name=path:sum`` metric specifications."""
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
