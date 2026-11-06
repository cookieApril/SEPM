"""ImplicitMemBench under an AutoGen-style team with pluggable memories (Table 3a, Fig. S2, Table S12).

Protocol (Supp. C.3):
* 300 items: 100 procedural memory (Proc), 100 classical conditioning (Cond), 100 priming (Prim);
* each item's learning/priming and interference turns form one exposure episode recorded
  into the memory; the team then answers the probe and only its first user-facing reply is
  scored with the benchmark's own judge (``gpt-4o-mini``) and rule-based checks;
* temperature 0 for Proc/Cond, 0.8 for the priming probe; one run, base seed 0;
* the memory pool is reset before every item unless ``--pool accumulate``.

Scores: Proc/Cond first-attempt accuracy (FTA, %), Prim priming influence score (PIS),
Overall = mean of the three.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
import sys
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ..domains import IMPLICIT_MEMORY_FAILURE_MARKERS
from ..hosts.autogen_team import AutoGenTeam
from ..llm_client import LLMClient, UsageMeter
from ..plugins.base import NoMemory

PROJECT_ROOT = Path(__file__).resolve().parents[3]
IMB_ROOT = PROJECT_ROOT / "external" / "ImplicitMemBench"
CATEGORIES = ("procedural_memory", "classical_conditioning", "priming")
SHORT = {"procedural_memory": "proc", "classical_conditioning": "cond", "priming": "prim"}
METHODS = ("single-llm", "no-memory", "generative", "metagpt", "gmemory", "mem0", "sepm")
JUDGE_MODEL = "gpt-4o-mini"
# The protocol allows 4096 output tokens; gpt-5-mini counts hidden reasoning against the cap,
# which left 15% of replies empty in a first run, so calls get 4x headroom.
EXECUTION_MAX_TOKENS = 16_384


def load_official_evaluation() -> Any:
    """Import the benchmark's evaluation module (judges, checks, conversation builders)."""
    path = IMB_ROOT / "src" / "evaluation" / "evaluation.py"
    spec = importlib.util.spec_from_file_location("imb_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(IMB_ROOT))
    spec.loader.exec_module(module)
    return module


def load_items() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for category in CATEGORIES:
        for path in sorted((IMB_ROOT / "dataset" / category).glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            for instance in data["instances"]:
                items.append(
                    {
                        "item_id": f"{SHORT[category]}/{path.stem}/{instance.get('task_id')}",
                        "category": category,
                        "task_seed": data.get("task_seed"),
                        "theme": data.get("source_theme") or instance.get("selected_source_theme"),
                        "instance": instance,
                    }
                )
    return items


def _normalize(phase: list[dict[str, Any]]) -> list[dict[str, str]]:
    messages = []
    for message in phase or []:
        if "role" in message and "content" in message:
            role, content = str(message["role"]).strip().lower(), str(message["content"]).strip()
        elif "speaker" in message and "text" in message:
            role = "user" if str(message["speaker"]).strip().lower() == "user" else "assistant"
            content = str(message["text"]).strip()
        else:
            continue
        if content:
            messages.append({"role": role if role in {"user", "assistant", "system"} else "assistant", "content": content})
    return messages


def _probe_text(probe: Any) -> str:
    if isinstance(probe, str):
        return probe.strip()
    if isinstance(probe, dict):
        return str(probe.get("content") or probe.get("text") or probe.get("prompt") or "").strip()
    if isinstance(probe, list):
        users = [m.get("content") or m.get("text") for m in probe if isinstance(m, dict)
                 and (m.get("role") == "user" or m.get("speaker") == "user")]
        return str(users[-1]).strip() if users else ""
    return ""


def _failed(observation: str) -> bool:
    lower = observation.lower()
    return any(marker in lower for marker in IMPLICIT_MEMORY_FAILURE_MARKERS)


def exposure_steps(messages: list[dict[str, str]]) -> list[tuple[str, str]]:
    """(assistant turn, what followed it) pairs: the observable steps of the exposure."""
    steps: list[tuple[str, str]] = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        following: list[str] = []
        for later in messages[index + 1 :]:
            if later["role"] == "assistant":
                break
            following.append(f"[{later['role']}] {later['content']}")
        steps.append((message["content"], "\n".join(following)))
    return steps


# ---------------------------------------------------------------------------
# Memory factory


class MemoryFactory:
    def __init__(self, method: str, client: LLMClient, *, model: str, work_dir: Path,
                 sepm_overrides: dict[str, Any] | None = None) -> None:
        self.method = method
        self.client = client
        self.model = model
        self.work_dir = work_dir
        self.sepm_overrides = sepm_overrides or {}
        if method in {"generative", "metagpt", "gmemory"}:
            from ..plugins._gmemory_compat import load_gmemory

            load_gmemory(PROJECT_ROOT / "external" / "GMemory")  # imports once, before worker threads

    def create(self, meter: UsageMeter, key: str) -> tuple[Any, Path | None]:
        if self.method in {"no-memory", "single-llm"}:
            return NoMemory(), None
        directory = Path(tempfile.mkdtemp(prefix=f"{self.method}-", dir=self.work_dir))
        if self.method == "sepm":
            from ..plugins.sepm_plugin import SEPMPlugin

            return SEPMPlugin(
                database_path=directory / "sepm.db", task_family="implicit_memory", client=self.client,
                meter=meter, model=self.model, judge_model=self.model,
                failure_markers=IMPLICIT_MEMORY_FAILURE_MARKERS, config_overrides=self.sepm_overrides,
            ), directory
        if self.method == "mem0":
            from ..plugins.mem0_plugin import Mem0Plugin

            return Mem0Plugin(client=self.client, meter=meter, model=self.model, path=directory, user_id=key), directory
        from ..plugins.gmemory_plugin import GMemoryFamilyPlugin

        return GMemoryFamilyPlugin(self.method, client=self.client, meter=meter, model=self.model,
                                   working_dir=directory), directory


# ---------------------------------------------------------------------------
# One context (an item, or one side of a priming pair)


def run_context(
    *,
    method: str,
    team: AutoGenTeam | None,
    client: LLMClient,
    model: str,
    memory: Any,
    meter: UsageMeter,
    key: str,
    exposure: list[dict[str, str]],
    probe: str,
    temperature: float,
) -> dict[str, Any]:
    if method == "single-llm":
        # evaluation._get_llm_response: two attempts when the reply is empty.
        for _ in range(2):
            reply = client.chat(model, exposure + [{"role": "user", "content": probe}], meter=meter,
                                component="execution", temperature=temperature, max_tokens=EXECUTION_MAX_TOKENS)
            if reply.strip():
                break
        return {"reply": reply}
    steps = exposure_steps(exposure)
    first_user = next((m["content"] for m in exposure if m["role"] == "user"), probe)
    memory.begin_task(f"{key}-exposure", f"Assist the user: {first_user[:500]}", "conversation with the user")
    for action, observation in steps:
        memory.record("assistant", action, observation)
    success = not (steps and _failed(steps[-1][1]))
    memory.end_task(success=success, reward=1.0 if success else 0.0)
    memory.begin_task(f"{key}-probe", probe[:2_000], "reply to the user's new message")
    turn = team.respond(exposure, probe, memory, temperature=temperature)
    retrieved = list(getattr(memory, "last_context_ids", []))
    memory.abandon_task()
    return {"reply": turn.reply, "drafts": turn.draft_count, "approved": turn.approved,
            "notices": turn.notices, "retrieved_ids": retrieved, "memory_stats": memory.stats()}


def run_item(item: dict[str, Any], *, method: str, factory: MemoryFactory, client: LLMClient,
             model: str, evaluation: Any, judge_client: Any, interference: dict[str, Any] | None,
             shared_memory: Any = None) -> dict[str, Any]:
    meter = UsageMeter()
    team = AutoGenTeam(client, meter, model=model) if method != "single-llm" else None
    instance = item["instance"]
    started = time.time()
    record: dict[str, Any] = {"item_id": item["item_id"], "category": item["category"], "method": method}
    cleanup: list[Path] = []

    def memory_for(suffix: str) -> Any:
        if shared_memory is not None:
            shared_memory.meter_override(meter)
            return shared_memory
        memory, directory = factory.create(meter, f"{item['item_id']}{suffix}".replace("/", "_"))
        if directory is not None:
            cleanup.append(directory)
        return memory

    try:
        if item["category"] == "priming":
            outputs = {}
            for side in ("experimental", "control"):
                context = instance.get(f"{side}_instance", {})
                exposure = _normalize(context.get("priming_phase", [])) + _normalize(context.get("interference_phase", []))
                probe = _probe_text(context.get("test_probe", {}))
                outputs[side] = run_context(method=method, team=team, client=client, model=model,
                                            memory=memory_for(f"-{side}"), meter=meter,
                                            key=f"{item['item_id']}-{side}", exposure=exposure,
                                            probe=probe, temperature=0.8)
            theme_map = evaluation._load_priming_theme_map()
            theme = item["theme"] or ""
            signature = dict(theme_map.get(theme, {"setting": "", "motifs": "", "dynamics": "", "affect": ""}))
            signature["name"] = theme
            exp_ctx, ctrl_ctx = instance.get("experimental_instance", {}), instance.get("control_instance", {})
            verdict = evaluation._judge_priming_pair(
                judge_client, JUDGE_MODEL, signature, _probe_text(exp_ctx.get("test_probe", {})),
                outputs["experimental"]["reply"], outputs["control"]["reply"],
                evaluation._extract_phase_text(exp_ctx.get("priming_phase", [])),
                evaluation._extract_phase_text(ctrl_ctx.get("priming_phase", [])),
            )
            if str(verdict.get("reasoning", "")).startswith("judge_error"):
                raise RuntimeError(f"priming judge failed: {verdict.get('reasoning')}")
            record.update(score=float(verdict["priming_influence_score"]), judge=verdict.get("reasoning", ""),
                          outputs=outputs)
        else:
            learning = _normalize(instance.get("learning_phase", []))
            interference_phase = _normalize(instance.get("interference_phase", []))
            if interference is not None:
                interference_phase = interference.get(item["item_id"], interference_phase)
            exposure = learning + interference_phase
            probe = _probe_text(instance.get("test_probe", {}))
            output = run_context(method=method, team=team, client=client, model=model,
                                 memory=memory_for(""), meter=meter, key=item["item_id"],
                                 exposure=exposure, probe=probe, temperature=0.0)
            rule_based = {"The Ternary Logic System", "The 'Omega' Operation", "The Modified Recurrence Sequence"}
            if item["category"] == "procedural_memory" and item["task_seed"] in rule_based:
                correct = evaluation._contains_expected(instance.get("expected_pattern"), output["reply"])
                reason = "rule-based expected_pattern inclusion"
            else:
                reason, correct = evaluation._judge_with_llm(judge_client, JUDGE_MODEL, instance,
                                                             output["reply"], item["category"])
            if str(reason).startswith("Judge error"):
                raise RuntimeError(f"judge failed: {reason}")
            record.update(score=100.0 if correct else 0.0, correct=bool(correct), judge=reason, outputs=output)
    finally:
        for directory in cleanup:
            shutil.rmtree(directory, ignore_errors=True)
    usage = meter.snapshot()
    record["usage"] = usage
    record["llm_calls"] = sum(v["calls"] for k, v in usage["components"].items() if k != "embedding")
    record["tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    record["seconds"] = round(time.time() - started, 1)
    return record


# ---------------------------------------------------------------------------
# Interference-length variants (Supp. C.3, D.4)

NEUTRAL_PROMPT = """Write {n} short, neutral, everyday question-and-answer exchanges between a user and an
assistant (general knowledge, cooking, travel, weather, simple math). They must NOT reuse any name,
term, symbol, rule, or topic from the following conversation:
---
{learning}
---
Return JSON {{"turns": [{{"user": str, "assistant": str}}, ...]}}."""


def _tokens(messages: list[dict[str, str]]) -> int:
    try:
        import tiktoken

        encoder = tiktoken.get_encoding("cl100k_base")
        return sum(len(encoder.encode(m["content"])) for m in messages)
    except Exception:  # noqa: BLE001 - fall back to a character estimate
        return sum(len(m["content"]) for m in messages) // 4


def build_interference(items: list[dict[str, Any]], target: int, client: LLMClient, cache: Path) -> dict[str, Any]:
    """~200 truncates the original interference; ~1000/2000 append neutral gpt-4o-mini turns."""
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    result: dict[str, Any] = {}
    for item in items:
        if item["category"] != "procedural_memory":
            continue
        original = _normalize(item["instance"].get("interference_phase", []))
        if target <= 200:
            kept: list[dict[str, str]] = []
            for message in original:
                if kept and _tokens(kept + [message]) > target:
                    break
                kept.append(message)
            if kept and kept[-1]["role"] == "user":
                kept = kept[:-1] or kept
            result[item["item_id"]] = kept
            continue
        learning = "\n".join(m["content"] for m in _normalize(item["instance"].get("learning_phase", [])))
        extended = list(original)
        attempts = 0
        while _tokens(extended) < target and attempts < 8:
            attempts += 1
            text = client.chat(JUDGE_MODEL, [{"role": "user", "content": NEUTRAL_PROMPT.format(n=6, learning=learning[:6_000])}],
                               temperature=0.7, max_tokens=2_000, json_mode=True)
            try:
                turns = json.loads(text[text.find("{"): text.rfind("}") + 1])["turns"]
            except (ValueError, KeyError):
                continue
            for turn in turns:
                if _tokens(extended) >= target:
                    break
                extended += [{"role": "user", "content": str(turn["user"])},
                             {"role": "assistant", "content": str(turn["assistant"])}]
        result[item["item_id"]] = extended
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# Driver


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if "score" in record:
            by_category[record["category"]].append(record["score"])
    scores = {SHORT[c]: (sum(v) / len(v) if v else None) for c, v in by_category.items()}
    present = [value for value in scores.values() if value is not None]
    n = len(records) or 1
    return {
        "n_items": {SHORT[c]: len(v) for c, v in by_category.items()},
        **scores,
        "overall": sum(present) / len(present) if len(present) == 3 else None,
        "tokens_per_item": sum(r.get("tokens", 0) for r in records) / n,
        "calls_per_item": sum(r.get("llm_calls", 0) for r in records) / n,
        "errors": sum(1 for r in records if "error" in r),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES), choices=CATEGORIES)
    parser.add_argument("--limit", type=int, help="items per category (pilot runs)")
    parser.add_argument("--pool", choices=("reset", "accumulate"), default="reset")
    parser.add_argument("--order", choices=("original", "shuffled"), default="original")
    parser.add_argument("--interference", type=int, choices=(200, 500, 1000, 2000), default=500)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "benchmark-results" / "implicitmem")
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)

    random.seed(args.seed)
    client = LLMClient(env_file=str(PROJECT_ROOT / ".env"))
    import os

    from openai import OpenAI

    judge_client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"], timeout=180, max_retries=6)
    evaluation = load_official_evaluation()

    items = [item for item in load_items() if item["category"] in args.categories]
    if args.limit:
        kept, seen = [], defaultdict(int)
        for item in items:
            if seen[item["category"]] < args.limit:
                kept.append(item)
                seen[item["category"]] += 1
        items = kept
    if args.order == "shuffled":
        random.Random(args.seed).shuffle(items)

    interference = None
    if args.interference != 500:
        interference = build_interference(
            items, args.interference, client, args.out / "interference" / f"proc_{args.interference}.json")

    name = args.tag or f"{args.method}__{args.pool}__{args.order}__int{args.interference}"
    out_dir = args.out / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"{name}.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if result_path.exists():
        for line in result_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if "error" not in row:
                    done[row["item_id"]] = row
    pending = [item for item in items if item["item_id"] not in done]
    print(f"[{name}] {len(done)} done, {len(pending)} pending", flush=True)

    work_dir = Path(tempfile.mkdtemp(prefix="imb-"))
    factory = MemoryFactory(args.method, client, model=args.model, work_dir=work_dir)
    lock = threading.Lock()

    def execute(item: dict[str, Any], shared: Any = None) -> dict[str, Any]:
        try:
            return run_item(item, method=args.method, factory=factory, client=client, model=args.model,
                            evaluation=evaluation, judge_client=judge_client, interference=interference,
                            shared_memory=shared)
        except Exception as error:  # noqa: BLE001 - keep the batch running; errors are rerun on resume
            return {"item_id": item["item_id"], "category": item["category"], "method": args.method,
                    "error": f"{type(error).__name__}: {error}"}

    def write(row: dict[str, Any]) -> None:
        with lock, result_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        status = "ERR " + row["error"][:120] if "error" in row else f"score={row.get('score')}"
        print(f"  {row['item_id']}: {status} calls={row.get('llm_calls')} t={row.get('seconds')}s", flush=True)

    try:
        if args.pool == "accumulate":

            if args.method != "sepm":
                raise SystemExit("--pool accumulate is defined for sepm only")
            # One persistent pool; items run sequentially so later items see earlier procedures.
            shared = AccumulatingSEPM(client, args.model, work_dir / "pool.db")
            for item in pending:
                write(execute(item, shared))
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(execute, item) for item in pending]
                for future in as_completed(futures):
                    write(future.result())
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    rows = {}
    for line in result_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if "error" not in row or row["item_id"] not in rows:
                rows[row["item_id"]] = row
    selected = [rows[item["item_id"]] for item in items if item["item_id"] in rows]
    summary = summarize([row for row in selected if "error" not in row])
    summary.update(method=args.method, pool=args.pool, order=args.order, interference=args.interference,
                   model=args.model, judge=JUDGE_MODEL, failed_items=sum(1 for r in selected if "error" in r))
    (out_dir / f"{name}.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


class AccumulatingSEPM:
    """One SEPM pool shared across items; tracks foreign procedures injected per item."""

    def __init__(self, client: LLMClient, model: str, database_path: Path) -> None:
        from ..plugins.sepm_plugin import SEPMPlugin

        self.meter = UsageMeter()
        self.plugin = SEPMPlugin(database_path=database_path, task_family="implicit_memory", client=client,
                                 meter=self.meter, model=model, judge_model=model,
                                 failure_markers=IMPLICIT_MEMORY_FAILURE_MARKERS)
        self.owner: dict[str, str] = {}
        self.current_item = ""

    def meter_override(self, meter: UsageMeter) -> None:
        # Charge this item's calls to the item meter.
        self.plugin.meter = meter
        self.plugin.embedder.meter = meter
        self.plugin.sop_model.meter = meter
        self.plugin.service.detector.goal_judge.meter = meter

    def _candidate_ids(self) -> set[str]:
        from ..models import CandidateStatus

        ids: set[str] = set()
        for status in CandidateStatus:
            rows, _ = self.plugin.service.store.list_candidates(status, limit=10_000, offset=0)
            ids.update(row.candidate_id for row in rows)
        active, _ = self.plugin.service.store.list_active_sops(limit=10_000, offset=0)
        ids.update(sop.sop_id for sop in active)
        return ids

    def begin_task(self, task_id: str, goal: str, description: str = "") -> None:
        self.current_item = task_id.rsplit("-", 1)[0]
        self._before = self._candidate_ids() if task_id.endswith("-exposure") else None
        self.plugin.begin_task(task_id, goal, description)

    def end_task(self, *, success: bool, reward: float) -> None:
        self.plugin.end_task(success=success, reward=reward)
        if self._before is not None:
            for candidate_id in self._candidate_ids() - self._before:
                self.owner[candidate_id] = self.current_item

    @property
    def last_context_ids(self) -> list[str]:
        ids = self.plugin.last_context_ids
        return [f"{i}|{'own' if self.owner.get(i) == self.current_item else 'foreign'}" for i in ids]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.plugin, name)


if __name__ == "__main__":
    raise SystemExit(main())
