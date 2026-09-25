"""论文表格只保留主性能、SOP 模型敏感性和组件消融。"""

from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from team_memory.paper_tables import (
    build_ablation_rows,
    build_main_rows,
    build_sop_model_rows,
    main,
)


def group(
    mas: str,
    memory: str,
    task: str,
    score: float,
    ablation: str = "full",
    sop_model: str = "not-applicable",
    benchmark: str = "cross-benchmark-generality",
    extra_metrics: dict[str, float] | None = None,
) -> dict:
    metric_stats = {
        "primary_score": {
            "mean": score,
            "ci95_low": score - 1,
            "ci95_high": score + 1,
            "n": 5,
        }
    }
    for name, value in (extra_metrics or {}).items():
        metric_stats[name] = {
            "mean": value,
            "ci95_low": value,
            "ci95_high": value,
            "n": 5,
        }
    return {
        "benchmark": benchmark,
        "task": task,
        "memory_method": memory,
        "actor_model": "actor-model",
        "sop_model": sop_model,
        "mas_framework": mas,
        "ablation": ablation,
        "successful_runs": 5,
        "metric_stats": metric_stats,
    }


class PaperTableTests(unittest.TestCase):
    def test_main_table_matches_cross_benchmark_layout(self) -> None:
        tasks = ["alfworld", "webarena"]
        report = {
            "groups": [
                group("autogen", "no-memory", "alfworld", 50),
                group("autogen", "no-memory", "webarena", 60),
                group("autogen", "agent-native-memory", "alfworld", 52),
                group("autogen", "agent-native-memory", "webarena", 62),
                group("autogen", "generative-memory", "alfworld", 55),
                group("autogen", "generative-memory", "webarena", 65),
                group("autogen", "gmemory", "alfworld", 65),
                group("autogen", "gmemory", "webarena", 70),
                group("autogen", "team-memory", "alfworld", 75, sop_model="sop-model"),
                group("autogen", "team-memory", "webarena", 85, sop_model="sop-model"),
            ]
        }
        rows = build_main_rows(
            report,
            actor_model="actor-model",
            sop_model="sop-model",
            tasks=tasks,
        )
        self.assertEqual(list(rows[0]), ["MAS", "Memory method", "ALFWorld", "WebArena", "Avg."])
        self.assertEqual([row["Memory method"] for row in rows], [
            "No-memory",
            "Agent-native memory",
            "Generative Memory",
            "G-Memory",
            "Team Memory (Ours)",
        ])
        ours = next(row for row in rows if row["Memory method"] == "Team Memory (Ours)")
        self.assertEqual(ours["ALFWorld"], 75)
        self.assertEqual(ours["Avg."], 80)

    def test_ablation_table_uses_only_minimal_conditions(self) -> None:
        tasks = ["alfworld"]
        report = {
            "groups": [
                group(
                    "autogen",
                    "team-memory",
                    "alfworld",
                    50,
                    "no-extra-components",
                    "sop-model",
                    benchmark="component-ablation",
                    extra_metrics={"case_count": 10},
                ),
                group(
                    "autogen",
                    "team-memory",
                    "alfworld",
                    55,
                    "blackboard-only",
                    "sop-model",
                    benchmark="component-ablation",
                ),
                group(
                    "autogen",
                    "team-memory",
                    "alfworld",
                    62,
                    "sop-only",
                    "sop-model",
                    benchmark="component-ablation",
                ),
                group(
                    "autogen",
                    "team-memory",
                    "alfworld",
                    58,
                    "divergence-only",
                    "sop-model",
                    benchmark="component-ablation",
                ),
                group(
                    "autogen",
                    "team-memory",
                    "alfworld",
                    70,
                    "full",
                    "sop-model",
                    benchmark="component-ablation",
                ),
            ]
        }
        rows = build_ablation_rows(
            report,
            actor_model="actor-model",
            sop_model="sop-model",
            mas_framework="autogen",
            tasks=tasks,
        )
        self.assertEqual(list(rows[0]), [
            "Benchmark",
            "Condition",
            "Procedural SOP",
            "Blackboard",
            "Divergence alignment",
            "Official task score",
            "Delta vs no-extra-components",
            "SOP reuse success",
            "Divergence recovery",
            "Unsafe accepted",
            "Token overhead",
            "Latency",
        ])
        self.assertEqual([row["Condition"] for row in rows], [
            "no-extra-components",
            "blackboard-only",
            "sop-only",
            "divergence-only",
            "full",
        ])
        full = next(row for row in rows if row["Condition"] == "full")
        self.assertEqual(full["Benchmark"], "ALFWorld")
        self.assertEqual(full["Procedural SOP"], "On")
        self.assertEqual(full["Divergence alignment"], "On")
        self.assertEqual(full["Delta vs no-extra-components"], 20)

    def test_sop_model_table_is_task_level_and_keeps_actor_fixed(self) -> None:
        report = {
            "groups": [
                group(
                    "autogen",
                    "team-memory",
                    "alfworld",
                    70,
                    sop_model="gpt-5-mini",
                    benchmark="sop-model-sensitivity",
                    extra_metrics={
                        "sop_json_valid_rate": 1.0,
                        "sop_candidate_pass_rate": 0.5,
                        "sop_prompt_tokens": 100,
                        "sop_completion_tokens": 20,
                    },
                ),
                group(
                    "autogen",
                    "team-memory",
                    "alfworld",
                    60,
                    sop_model="qwen3.5-9b",
                    benchmark="sop-model-sensitivity",
                ),
            ]
        }
        rows = build_sop_model_rows(
            report,
            actor_model="actor-model",
            default_sop_model="gpt-5-mini",
            tasks=["alfworld"],
        )
        self.assertEqual({row["SOP model"] for row in rows}, {"gpt-5-mini", "qwen3.5-9b"})
        qwen = next(row for row in rows if row["SOP model"] == "qwen3.5-9b")
        self.assertEqual(qwen["Delta vs default SOP model"], -10)
        default = next(row for row in rows if row["SOP model"] == "gpt-5-mini")
        self.assertEqual(default["SOP-model cost"], 120)

    def test_cli_writes_only_three_paper_tables(self) -> None:
        report = {
            "groups": [
                group("autogen", "no-memory", "alfworld", 50),
                group("autogen", "team-memory", "alfworld", 70, sop_model="sop-model"),
                group(
                    "autogen",
                    "team-memory",
                    "alfworld",
                    70,
                    sop_model="sop-model",
                    benchmark="sop-model-sensitivity",
                ),
            ]
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "summary.json"
            summary.write_text(json.dumps(report), encoding="utf-8")
            output_dir = root / "tables"

            exit_code = main(
                [
                    "--summary",
                    str(summary),
                    "--actor-model",
                    "actor-model",
                    "--sop-model",
                    "sop-model",
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            files = {path.name for path in output_dir.iterdir()}
            self.assertEqual(files, {
                "cross_benchmark_main.csv",
                "sop_model_sensitivity.csv",
                "ablation.csv",
            })
            with (output_dir / "cross_benchmark_main.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["Memory method"], "No-memory")


if __name__ == "__main__":
    unittest.main()
