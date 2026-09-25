"""Extract the two matched E3 cases discussed in the appendix.

The output contains source paths, official outcomes, trajectories when retained by
the benchmark adapter, and runtime counters.  It does not infer causal attribution.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmark-results" / "unified-v3" / "results"
OUTPUT = Path(__file__).resolve().parent / "data" / "E3-case-study.json"

SELECTIONS = (
    {
        "benchmark": "alfworld",
        "case_contains": "pick_heat_then_place_in_recep-Egg-None-GarbageCan-10",
        "display_case": "heat-egg-to-garbagecan",
    },
    {
        "benchmark": "officebench",
        "case_contains": "2-39/1",
        "display_case": "2-39/1",
    },
)


def load_records(selection: dict[str, str]) -> list[dict]:
    records = []
    pattern = f"component-ablation__{selection['benchmark']}__*.json"
    for path in RESULTS.glob(pattern):
        if path.name.endswith("team-memory-runtime-metrics.json"):
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        context = record.get("context", {})
        if context.get("seed") != 0:
            continue
        if selection["case_contains"] not in context.get("case_id", ""):
            continue
        if context.get("ablation") not in {"no-extra-components", "full"}:
            continue
        case = record["cases"][0]
        records.append(
            {
                "condition": context["ablation"],
                "source": path.relative_to(ROOT).as_posix(),
                "official_score": record["metrics"]["primary_score"],
                "goal_or_instruction": case.get("task", {}).get("goal", case.get("instruction")),
                "trajectory": [
                    {
                        "step": step["step"],
                        "action": step["action"],
                        "observation": step["observation"],
                        "reward": step["reward"],
                        "done": step["done"],
                        "parser_recovery_used": step["parser_recovery_used"],
                    }
                    for step in case.get("trajectory", [])
                ],
                "evaluation_functions": case.get("evaluation_functions", []),
                "blackboard_entries": record["metrics"].get("blackboard_entries"),
                "world_state_conflict_count": record["metrics"].get("world_state_conflict_count"),
                "divergence_recovery_count": record["metrics"].get("divergence_recovery"),
                "sop_retrieval_count": record["metrics"].get("sop_retrieval_count"),
            }
        )
    return sorted(records, key=lambda item: item["condition"])


def main() -> None:
    output = []
    for selection in SELECTIONS:
        records = load_records(selection)
        if {record["condition"] for record in records} != {"full", "no-extra-components"}:
            raise RuntimeError(f"Incomplete matched pair for {selection['display_case']}")
        output.append({"case": selection["display_case"], "records": records})
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
