"""Aggregate experiment outputs into the paper's tables (markdown + JSON).

Reads whatever exists; missing conditions are shown as "–" rather than filled in.

* ImplicitMemBench (E4/E5): ``benchmark-results/implicitmem/<model>/*.jsonl``
* MultiAgentBench (E3):     ``benchmark-results/multiagentbench/<scenario>/<method>/results.jsonl``
* Linux runs (E1, E6, E8-E10, E12, E13): ``--linux benchmark-results/linux`` (adapter JSONs)
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
IMB_METHODS = [("single-llm", "Single LLM (no MAS)"), ("no-memory", "AutoGen, no added memory"),
               ("generative", "+ Generative Agents"), ("metagpt", "+ MetaGPT"), ("gmemory", "+ G-Memory"),
               ("mem0", "+ Mem0 (automatic)"), ("sepm", "+ SEPM")]
MAB_METHODS = [("no-memory", "No added memory"), ("generative", "Generative"), ("metagpt", "MetaGPT"),
               ("gmemory", "G-Memory"), ("sepm", "SEPM")]
E1_METHODS = [("no-memory", "No added memory"), ("generative", "Generative"), ("metagpt", "MetaGPT"),
              ("gmemory", "G-Memory"), ("sepm", "SEPM")]


def _fmt(value: Any, digits: int = 2) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            key = row.get("item_id") or row.get("case_id")
            if "error" not in row or key not in rows:
                rows[key] = row
    return list(rows.values())


# ---------------------------------------------------------------------------
# ImplicitMemBench


def imb_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if "error" not in r and "score" in r]
    by: dict[str, list[float]] = defaultdict(list)
    for row in ok:
        by[row["category"]].append(row["score"])
    proc, cond, prim = (sum(by[c]) / len(by[c]) if by[c] else None
                        for c in ("procedural_memory", "classical_conditioning", "priming"))
    overall = (proc + cond + prim) / 3 if None not in (proc, cond, prim) else None
    n = len(ok) or 1
    return {"proc": proc, "cond": cond, "prim": prim, "overall": overall,
            "n": {c: len(v) for c, v in by.items()}, "errors": len(rows) - len(ok),
            "tokens_per_item": sum(r.get("tokens", 0) for r in ok) / n,
            "calls_per_item": sum(r.get("llm_calls", 0) for r in ok) / n}


def report_implicitmem(root: Path, model: str = "gpt-5-mini") -> tuple[str, dict[str, Any]]:
    base = root / model
    data: dict[str, Any] = {}
    out = [f"### ImplicitMemBench ({model}; Table 7 / Fig. 5)\n"]
    scores = {m: imb_scores(_jsonl(base / f"{m}__reset__original__int500.jsonl")) for m, _ in IMB_METHODS}
    host = scores["no-memory"]["overall"]
    rows = []
    for method, label in IMB_METHODS:
        s = scores[method]
        delta = None if s["overall"] is None or host is None or method == "no-memory" else s["overall"] - host
        rows.append([label, _fmt(s["proc"]), _fmt(s["cond"]), _fmt(s["prim"]), _fmt(s["overall"]),
                     "–" if delta is None else f"{delta:+.2f}",
                     f"{s['tokens_per_item'] / 1000:.1f}k", f"{s['calls_per_item']:.1f}",
                     "/".join(str(s["n"].get(c, 0)) for c in ("procedural_memory", "classical_conditioning", "priming"))])
    out.append(_table(["Method", "Proc.", "Cond.", "Prim.", "Overall", "Δ", "Tokens/item", "Calls/item", "n (P/C/Pr)"], rows))
    data["main"] = scores

    rows = []
    for method in ("single-llm", "no-memory", "gmemory", "sepm"):
        row = [method]
        for n in (200, 500, 1000, 2000):
            rows_n = _jsonl(base / f"{method}__reset__original__int{n}.jsonl")
            proc = [r["score"] for r in rows_n if r.get("category") == "procedural_memory" and "score" in r]
            row.append(f"{sum(proc) / len(proc):.0f} (n={len(proc)})" if proc else "–")
        rows.append(row)
    out.append("\n#### Proc. FTA vs interference length (Fig. 8)\n")
    out.append(_table(["Method", "~200", "~500 (orig.)", "~1000", "~2000"], rows))

    reset = {r["item_id"]: r for r in _jsonl(base / "sepm__reset__original__int500.jsonl") if "score" in r}
    rows = [["Reset (main)", *(_fmt(scores["sepm"][k]) for k in ("proc", "cond", "prim", "overall")), "0", "–", "–"]]
    for order in ("original", "shuffled"):
        acc = _jsonl(base / f"sepm__accumulate__{order}__int500.jsonl")
        s = imb_scores(acc)
        foreign = []
        for row in acc:
            outputs = row.get("outputs", {})
            parts = [outputs] if "reply" in outputs else list(outputs.values())
            foreign.append(sum(1 for part in parts for i in part.get("retrieved_ids", []) if i.endswith("|foreign")))
        neg = pos = 0
        for row in acc:
            if row.get("category") in ("procedural_memory", "classical_conditioning") and row["item_id"] in reset and "score" in row:
                before, after = reset[row["item_id"]]["score"] > 0, row["score"] > 0
                neg += before and not after
                pos += after and not before
        rows.append([f"Accumulate, {order}", *(_fmt(s[k]) for k in ("proc", "cond", "prim", "overall")),
                     _fmt(sum(foreign) / len(foreign)) if foreign else "–", str(neg) if acc else "–", str(pos) if acc else "–"])
    out.append("\n#### Pool reset vs accumulate (Table 8)\n")
    out.append(_table(["Pool", "Proc.", "Cond.", "Prim.", "Overall", "Foreign/item", "Neg.", "Pos."], rows))
    return "\n".join(out), data


# ---------------------------------------------------------------------------
# MultiAgentBench


def report_mab(root: Path) -> tuple[str, dict[str, Any]]:
    data: dict[str, Any] = {}
    header = ["Method"]
    scenarios = ("coding", "research", "database")
    for scenario in scenarios:
        header += [f"{scenario.title()} TS", f"{scenario.title()} CS", "n"]
    rows = []
    for method, label in MAB_METHODS:
        row = [label]
        for scenario in scenarios:
            results = [r for r in _jsonl(root / scenario / method / "results.jsonl") if "error" not in r]
            ts = [r["ts"] for r in results if r.get("ts") is not None]
            cs = [r["cs"] for r in results if r.get("cs") is not None]
            data[f"{scenario}/{method}"] = {"ts": sum(ts) / len(ts) if ts else None,
                                            "cs": sum(cs) / len(cs) if cs else None, "n": len(results)}
            row += [_fmt(data[f"{scenario}/{method}"]["ts"]), _fmt(data[f"{scenario}/{method}"]["cs"]), str(len(results))]
        rows.append(row)
    return "### MultiAgentBench (Table 6 / Fig. 1)\n\n" + _table(header, rows), data


# ---------------------------------------------------------------------------
# Linux adapter outputs


def _load_runs(root: Path, group: str) -> dict[str, dict[str, Any]]:
    runs = {}
    for path in sorted((root / group).glob("*.json")):
        try:
            runs[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
    return runs


def _successes(run: dict[str, Any]) -> dict[str, float]:
    return {str(case["case_id"]): (1.0 if case.get("done") else 0.0) for case in run.get("cases", [])}


def sign_flip_test(diffs: list[float], permutations: int = 200_000, seed: int = 0) -> float:
    """Two-sided paired sign-flip permutation test on per-task differences."""
    observed = abs(sum(diffs))
    nonzero = [d for d in diffs if d]
    if not nonzero:
        return 1.0
    rng = random.Random(seed)
    extreme = sum(abs(sum(d if rng.random() < 0.5 else -d for d in nonzero)) >= observed - 1e-12
                  for _ in range(permutations))
    return (extreme + 1) / (permutations + 1)


def report_linux(root: Path) -> tuple[str, dict[str, Any]]:
    out, data = [], {}
    e1 = _load_runs(root, "e1")
    pattern = re.compile(r"^(?P<task>[^_]+)__(?P<mas>[^_]+)__(?P<method>[a-z0-9-]+)__(?P<abl>[a-z-]+)__s(?P<seed>\d+)$")
    grouped: dict[tuple[str, str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for name, run in e1.items():
        m = pattern.match(name)
        if m:
            grouped[(m["task"], m["mas"], m["method"])][int(m["seed"])] = run
    rows = []
    for mas in ("autogen", "dylan"):
        for method, label in E1_METHODS:
            row = [mas, label]
            for task in ("alfworld", "officebench"):
                seeds = grouped.get((task, mas, method), {})
                values = [run["metrics"]["primary_score"] * 100 for run in seeds.values()]
                row.append(f"{sum(values) / len(values):.2f} (seeds {sorted(seeds)})" if values else "–")
                data[f"e1/{task}/{mas}/{method}"] = values
            rows.append(row)
    out.append("### E1 task score (%, Table 1; ALFWorld/OfficeBench)\n\n" + _table(["Host", "Memory", "ALFWorld", "OfficeBench"], rows))

    # Significance vs no added memory (needs the same seeds for both methods).
    rows = []
    for mas in ("autogen", "dylan"):
        for task in ("alfworld", "officebench"):
            base, sepm = grouped.get((task, mas, "no-memory"), {}), grouped.get((task, mas, "sepm"), {})
            seeds = sorted(set(base) & set(sepm))
            if not seeds:
                continue
            totals_b, totals_s = defaultdict(float), defaultdict(float)
            for seed in seeds:
                for case, value in _successes(base[seed]).items():
                    totals_b[case] += value
                for case, value in _successes(sepm[seed]).items():
                    totals_s[case] += value
            cases = sorted(set(totals_b) & set(totals_s))
            diffs = [totals_s[c] - totals_b[c] for c in cases]
            wins, ties, losses = sum(d > 0 for d in diffs), sum(d == 0 for d in diffs), sum(d < 0 for d in diffs)
            delta = 100 * sum(diffs) / (len(cases) * len(seeds)) if cases else 0.0
            rows.append([mas, task, str(seeds), f"{delta:+.2f}", f"{wins}/{ties}/{losses}", f"{sign_flip_test(diffs):.2g}"])
    if rows:
        out.append("\n### SEPM vs no added memory (Table 4)\n\n" + _table(["Host", "Benchmark", "Seeds", "Δ (pts)", "W/T/L", "p"], rows))

    # Learning curves (Fig. 6) and cost (Tables 13, 14) from seed-0 AutoGen runs.
    rows, cost_rows = [], []
    for task in ("alfworld", "officebench"):
        for method, label in E1_METHODS:
            run = grouped.get((task, "autogen", method), {}).get(0)
            if not run:
                continue
            cases = run.get("cases", [])
            k = len(cases)
            segments = [cases[round(i * k / 5): round((i + 1) * k / 5)] for i in range(5)]
            rows.append([task, label] + [_fmt(100 * sum(c.get("done", False) for c in seg) / len(seg)) if seg else "–" for seg in segments])
            m = run["metrics"]
            tokens = m.get("actor_prompt_tokens", 0) + m.get("actor_completion_tokens", 0) + m.get("sop_prompt_tokens", 0) + m.get("sop_completion_tokens", 0)
            seconds = [c.get("seconds") for c in cases if c.get("seconds") is not None]
            calls = m.get("actor_calls", 0) + m.get("sop_model_calls", 0)
            cost_rows.append([task, label, f"{tokens / 1e6:.2f}", f"{calls / max(1, k):.1f}",
                              f"{sum(seconds) / len(seconds):.0f}" if seconds else "–"])
    if rows:
        out.append("\n### Learning curve by segment, AutoGen seed 0 (Fig. 6)\n\n" + _table(["Benchmark", "Memory", "B1", "B2", "B3", "B4", "B5"], rows))
        out.append("\n### Cost, AutoGen seed 0 (Table 13)\n\n" + _table(["Benchmark", "Memory", "Tokens (M)", "Calls/task", "s/task"], cost_rows))

    # Pool statistics and realignment (Tables 9, 10).
    rows = []
    for task in ("alfworld", "officebench"):
        run = grouped.get((task, "autogen", "sepm"), {}).get(0)
        if not run:
            continue
        cases = run.get("cases", [])
        realigned = [c for c in cases if c.get("counters", {}).get("alignment_events", 0) > 0]
        others = [c for c in cases if c not in realigned]
        counters = [c.get("counters", {}) for c in cases]
        n = len(cases) or 1
        rows.append([task,
                     _fmt(sum(c.get("alignment_goal_divergence", 0) for c in counters) / n),
                     _fmt(sum(c.get("alignment_plan_divergence", 0) for c in counters) / n),
                     _fmt(sum(c.get("alignment_world_state_conflict", 0) for c in counters) / n),
                     f"{100 * len(realigned) / n:.2f}% ({len(realigned)}/{len(cases)})",
                     f"{100 * sum(c.get('done', False) for c in realigned) / max(1, len(realigned)):.2f}%",
                     f"{100 * sum(c.get('done', False) for c in others) / max(1, len(others)):.2f}%",
                     json.dumps(run.get("metadata", {}).get("pool_statistics"))])
    if rows:
        out.append("\n### Realignment and pool statistics, AutoGen seed 0 (Tables 9, 10)\n\n" + _table(
            ["Benchmark", "Goal/task", "Plan/task", "State/task", "Tasks w/ realign.", "Succ. w/", "Succ. w/o", "Pool"], rows))

    for group, title in (("ablation", "Component ablation (Fig. 7, Table 12)"), ("sensitivity", "Hyperparameter sensitivity (Fig. 10)"),
                         ("teamsize", "Team size (Table 11)"), ("loo-test", "Leave-one-type-out, frozen (Fig. 9a)"),
                         ("transfer", "Frozen AutoGen pool under DyLAN (Fig. 9b)")):
        runs = _load_runs(root, group)
        if runs:
            rows = [[name, _fmt(run["metrics"].get("primary_score", 0) * 100), str(len(run.get("cases", [])))] for name, run in runs.items()]
            out.append(f"\n### {title}\n\n" + _table(["Run", "Score (%)", "n"], rows))
    return "\n".join(out), data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implicitmem", type=Path, default=PROJECT_ROOT / "benchmark-results" / "implicitmem")
    parser.add_argument("--mab", type=Path, default=PROJECT_ROOT / "benchmark-results" / "multiagentbench")
    parser.add_argument("--linux", type=Path, default=PROJECT_ROOT / "benchmark-results" / "linux")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "benchmark-results" / "REPORT.md")
    args = parser.parse_args(argv)
    sections, data = [], {}
    for name, fn, path in (("implicitmem", report_implicitmem, args.implicitmem), ("mab", report_mab, args.mab),
                           ("linux", report_linux, args.linux)):
        if path.exists():
            text, values = fn(path)
            sections.append(text)
            data[name] = values
    args.out.write_text("# SEPM re-run results\n\n" + "\n\n".join(sections) + "\n", encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(args.out.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
