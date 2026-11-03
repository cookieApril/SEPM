"""MultiAgentBench (MARBLE) coding/research with pluggable memories (E3).

Protocol (paper Appendix C.3, MultiAgentBench paragraph):
* 100 tasks per scenario; task definitions, agent profiles, environment rules and the
  communication graph come from the official MARBLE configs (graph coordination);
* at most three iterations per task; ``gpt-5-mini`` executes, ``claude-opus-5`` evaluates;
* generation temperature 0.7 with at most 1024 output tokens (coding: 0.1 and 4096); because
  gpt-5-mini's reasoning tokens count toward the cap, calls get ``REASONING_HEADROOM`` x that budget;
* memory persists across the tasks of one condition.

Host note: the official MARBLE engine is the host here; memories plug into each
agent's ``BaseMemory`` (the slot MARBLE inserts into act/plan prompts and updates after
every action), so coordination, tools, and communication stay MARBLE's own.

Scores (all MARBLE ratings are on 1-5 and reported x20 on 0-100):
* TS research = mean(innovation, safety, feasibility); TS coding = mean of MARBLE's
  ``evaluate_code_quality`` criteria on the produced ``solution.py``;
* CS = mean of the per-iteration communication and planning ratings from MARBLE's
  ``evaluate_communication``/``evaluate_planning`` prompts (disabled in the released
  graph loop, called here after each task on the recorded iterations).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..llm_client import CachedEmbedder, LLMClient, UsageMeter

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MARBLE_ROOT = PROJECT_ROOT / "external" / "MARBLE"
ACTOR_MODEL = "gpt-5-mini"
JUDGE_MODEL = "claude-opus-5"
JUDGE_ALIAS = "sepm-judge"
METHODS = ("no-memory", "generative", "metagpt", "gmemory", "sepm")
SCENARIO_SETTINGS = {
    "research": {"temperature": 0.7, "max_tokens": 1024},
    "coding": {"temperature": 0.1, "max_tokens": 4096},
    "database": {"temperature": 0.7, "max_tokens": 1024},
}
# gpt-5-mini counts hidden reasoning tokens against max_completion_tokens; with the protocol's
# 1024/4096 caps alone, code generation returned empty files. The protocol limits are applied
# to the visible output by giving reasoning this much headroom.
REASONING_HEADROOM = 4
# The DB environment brings up one docker-compose Postgres on fixed ports: one task at a time.
_DB_LOCK = threading.Lock()
DB_LABEL_MODEL = "gpt-4o-mini"  # scripts/database/batch_eval.py label extraction
MARBLE_FAILURE_MARKERS = ("'success': false", '"success": false', "error-msg", "traceback", "error:")


# ---------------------------------------------------------------------------
# LLM routing: every MARBLE call goes through one metered client


class Router:
    def __init__(self, client: LLMClient, scenario: str) -> None:
        self.client = client
        self.settings = SCENARIO_SETTINGS[scenario]
        self.meter = UsageMeter()
        self.empty_responses = 0
        self.truncated_tool_calls = 0

    def __call__(self, llm_model: str, messages: list[dict[str, Any]], return_num: int | None = 1,
                 max_token_num: int | None = 512, temperature: float | None = 0.0, top_p: float | None = None,
                 stream: bool | None = None, mode: str | None = None, tools: list[dict[str, Any]] | None = None,
                 tool_choice: str | None = None) -> list[Any]:
        judge = llm_model == JUDGE_ALIAS
        if judge:
            model, temp, limit, component = JUDGE_MODEL, temperature, max_token_num or 512, "judge"
        else:
            # Agents may name any model in tool arguments; the actor is fixed by the protocol.
            model = ACTOR_MODEL
            temp = self.settings["temperature"]
            limit = self.settings["max_tokens"] * REASONING_HEADROOM
            component = "execution"
        message = self.client.complete(
            model, _clean_messages(messages), meter=self.meter, component=component, temperature=temp,
            max_tokens=limit, tools=_clean_tools(tools), tool_choice=tool_choice if tools else None,
        )
        if not judge and not (message.content or message.tool_calls):
            self.empty_responses += 1
        for call in message.tool_calls or []:
            try:
                json.loads(call.function.arguments or "{}")
            except ValueError:
                self.truncated_tool_calls += 1  # MARBLE's agent then logs an error and moves on
        return [message]


def _clean_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = []
    for message in messages:
        role = message.get("role", "user")
        cleaned.append({"role": role if role in {"system", "user", "assistant"} else "user",
                        "content": str(message.get("content") or "")})
    return cleaned


def _clean_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Repair MARBLE tool schemas that strict OpenAI-compatible endpoints reject.

    Empty ``enum`` lists are dropped, and ``required`` keeps only declared properties
    (``communicate_to`` requires ``target_agent_id`` without declaring it).
    """
    if not tools:
        return None
    payload = json.loads(json.dumps(tools))
    for tool in payload:
        parameters = tool.get("function", {}).get("parameters", {})
        properties = parameters.get("properties", {})
        for prop in properties.values():
            if isinstance(prop, dict) and prop.get("enum") == []:
                prop.pop("enum")
        if isinstance(parameters.get("required"), list):
            parameters["required"] = [name for name in parameters["required"] if name in properties]
    return payload


_CTX = threading.local()


def _dispatch(*args: Any, **kwargs: Any) -> list[Any]:
    """MARBLE's ``model_prompting`` replacement: route to the calling condition's router."""
    return _CTX.router(*args, **kwargs)


def _case_path(kind: str) -> str:
    case_dir: Path = _CTX.case_dir
    return str(case_dir / {"config": "marble/configs/coding_config/coding_config.yaml",
                           "solution": "marble/workspace/solution.py"}[kind])


def _rewrite_case_paths(module: Any) -> None:
    """Re-execute a MARBLE module with its fixed relative case paths bound to the thread's case."""
    path = Path(module.__file__)
    source = path.read_text(encoding="utf-8")
    source = source.replace('"marble/configs/coding_config/coding_config.yaml"', '_sepm_case_path("config")')
    source = source.replace('"marble/workspace/solution.py"', '_sepm_case_path("solution")')
    module.__dict__["_sepm_case_path"] = _case_path
    exec(compile(source, str(path), "exec"), module.__dict__)  # noqa: S102 - trusted local checkout


def install_marble(embedder: CachedEmbedder) -> dict[str, Any]:
    """Import MARBLE once per process with the routing, path, and KeyBERT adapters."""
    from ._keybert_compat import install_keybert

    install_keybert(embedder)
    sys.path.insert(0, str(MARBLE_ROOT))
    # MARBLE reads evaluator/evaluator_prompts.json and logs/ relative to marble/.
    os.chdir(MARBLE_ROOT / "marble")
    (MARBLE_ROOT / "marble" / "logs").mkdir(exist_ok=True)
    import marble.agent.base_agent as base_agent
    import marble.environments.coding_utils.coder as coder
    import marble.environments.coding_utils.reviewer as reviewer
    import marble.evaluator.evaluator as evaluator
    import marble.llms.model_prompting  # noqa: F401 - the package re-exports the function under this name
    from marble.configs.config import Config
    from marble.engine.engine import Engine
    from marble.memory.base_memory import BaseMemory

    prompting = sys.modules["marble.llms.model_prompting"]
    original = prompting.model_prompting
    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("marble") and getattr(module, "model_prompting", None) is original:
            module.model_prompting = _dispatch
    prompting.model_prompting = _dispatch
    for module in (coder, reviewer, evaluator):
        _rewrite_case_paths(module)
        module.model_prompting = _dispatch
    return {"Config": Config, "Engine": Engine, "Evaluator": evaluator.Evaluator, "BaseMemory": BaseMemory,
            "BaseAgent": base_agent.BaseAgent}


def plugin_memory_class(base_memory: type) -> type:
    class PluginMemory(base_memory):
        """MARBLE agent memory backed by a SEPM-style plug-in."""

        def __init__(self, agent: Any, plugin: Any, task: str) -> None:
            super().__init__()
            self.agent = agent
            self.plugin = plugin
            self.task = task

        def update(self, key: str, information: Any) -> None:
            super().update(key, information)
            if not isinstance(information, dict):
                return
            if information.get("type") == "action_function_call":
                action = f"{information.get('action_name')}: {json.dumps(information.get('args'), ensure_ascii=False)[:1500]}"
                observation = str(information.get("result"))[:3000]
            else:
                action, observation = str(information.get("result"))[:3000], ""
            goal = getattr(self.agent, "_sepm_current_task", None)
            self.plugin.record(self.agent.agent_id, action, observation, goal=goal)

        def get_memory_str(self) -> str:
            own = super().get_memory_str()
            context = self.plugin.context(self.agent.agent_id, self.task)
            notices = self.plugin.notices(self.agent.agent_id)
            parts = [own]
            if context:
                parts.append(f"\nTeam memory:\n{context}")
            if notices:
                parts.append("\nTeam alignment notices:\n" + "\n".join(f"- {item}" for item in notices))
            return "\n".join(part for part in parts if part)

    return PluginMemory


# ---------------------------------------------------------------------------
# Cases


def load_cases(scenario: str) -> list[dict[str, Any]]:
    import yaml

    cases = []
    if scenario == "coding":
        for task_id in range(1, 101):
            path = MARBLE_ROOT / "marble" / "configs" / "coding_configs" / f"config_{task_id}.yaml"
            cases.append({"case_id": f"coding/{task_id}", "config": yaml.safe_load(path.read_text(encoding="utf-8"))})
        return cases
    environment = {
        "research": {"max_iterations": 3, "name": "Research Collaboration Environment", "type": "Research"},
        "database": {"max_iterations": 3, "name": "DB Environment", "type": "DB"},
    }[scenario]
    defaults = {
        "coordinate_mode": "graph", "llm": ACTOR_MODEL,
        "environment": environment,
        "memory": {"type": "BaseMemory"}, "metrics": {"evaluate_llm": JUDGE_ALIAS},
        "output": {"file_path": "result/discussion_output.jsonl"},
    }
    path = MARBLE_ROOT / "multiagentbench" / scenario / f"{scenario}_main.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        # Official multiagentbench/jsonl2yaml.py fill rule: replace empty values only.
        for key in ("coordinate_mode", "llm"):
            if data.get(key, None) == "":
                data[key] = defaults[key]
        for key in ("environment", "memory", "output"):
            if isinstance(data.get(key), dict):
                for sub, value in defaults[key].items():
                    if data[key].get(sub) == "":
                        data[key][sub] = value
        if isinstance(data.get("metrics"), dict) and data["metrics"].get("evaluate_llm") == "":
            data["metrics"]["evaluate_llm"] = defaults["metrics"]["evaluate_llm"]
        cases.append({"case_id": f"{scenario}/{data.get('task_id')}", "config": data})
    return cases


def make_plugin(method: str, client: LLMClient, meter: UsageMeter, pool_dir: Path, scenario: str) -> Any:
    from ..plugins.base import NoMemory

    pool_dir.mkdir(parents=True, exist_ok=True)
    if method == "no-memory":
        return NoMemory()
    if method == "sepm":
        from ..plugins.sepm_plugin import SEPMPlugin

        return SEPMPlugin(database_path=pool_dir / "sepm.db", task_family=f"marble_{scenario}", client=client,
                          meter=meter, model=ACTOR_MODEL, judge_model=ACTOR_MODEL,
                          failure_markers=MARBLE_FAILURE_MARKERS)
    from ..plugins.gmemory_plugin import GMemoryFamilyPlugin

    return GMemoryFamilyPlugin(method, client=client, meter=meter, model=ACTOR_MODEL,
                               gmemory_root=PROJECT_ROOT / "external" / "GMemory", working_dir=pool_dir)


def _graph_rating(evaluator: Any, kind: str, fields: dict[str, str]) -> int:
    """MARBLE's Graph Communication/Planning rating with its own prompt and ``parse_score``.

    The released templates contain literal JSON braces (``{"rating": X}``), so the
    upstream ``str.format`` call raises KeyError; only the named fields are substituted.
    """
    prompt = evaluator.evaluation_prompts["Graph"][kind]["prompt"]
    for key, value in fields.items():
        prompt = prompt.replace("{" + key + "}", value)
    result = _CTX.router(JUDGE_ALIAS, [{"role": "user", "content": prompt}], max_token_num=512, temperature=0.0)[0]
    return evaluator.parse_score(result.content or "")


CODE_KEYS = ("instruction_following", "executability", "consistency", "quality")


def _parse_code_quality(text: str) -> dict[str, int] | None:
    """MARBLE's code-quality scores, tolerant of extra text or a second JSON block.

    The released parser takes everything between the first '{' and the last '}', so any
    trailing JSON or comment makes it fail and silently score 1 on every criterion.
    """
    import re

    start = text.find("{")
    if start >= 0:
        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(value, dict) and all(key in value for key in CODE_KEYS):
                return {key: min(5, max(1, int(value[key]))) for key in CODE_KEYS}
        except (ValueError, TypeError):
            pass
    found = {key: re.search(rf"[\"']?{key}[\"']?\s*:\s*([1-5])", text) for key in CODE_KEYS}
    if all(found.values()):
        return {key: int(match.group(1)) for key, match in found.items()}
    return None


def _code_quality(evaluator: Any, task_text: str, solution: Path, attempts: int = 3) -> dict[str, Any]:
    """Run MARBLE's ``evaluate_code_quality`` with a robust parser; re-ask the judge on parse failure."""
    raw: list[str] = []

    def parse(text: str) -> dict[str, int]:
        raw.append(text or "")
        scores = _parse_code_quality(text or "")
        if scores is None:
            raise ValueError("unparseable code-quality verdict")
        return scores

    evaluator.parse_code_quality_scores = parse
    code = solution.read_text(encoding="utf-8", errors="replace")
    for attempt in range(1, attempts + 1):
        raw.clear()
        evaluator.metrics["code_quality"] = {}
        evaluator.evaluate_code_quality(task=task_text, code_result=code)
        scores = evaluator.metrics.get("code_quality") or {}
        # On a parse error MARBLE stores its all-1 default; a parsed verdict always passed parse().
        if raw and _parse_code_quality(raw[-1]) is not None:
            return {**scores, "judge_attempts": attempt}
    raise RuntimeError(f"code-quality judge returned no parseable verdict in {attempts} attempts: {raw[-1:]!r}")


def _database_task_score(router: Router, task_evaluation: dict[str, Any]) -> dict[str, Any]:
    """scripts/database/batch_eval.py: gpt-4o-mini extracts two labels; score = matched gold fraction."""
    gold = [str(label).strip() for label in task_evaluation.get("root_cause", [])]
    predicted = str(task_evaluation.get("predicted", ""))
    if not gold:
        return {"fraction": None, "gold": gold, "predicted_labels": []}
    prompt = (
        f"{predicted}\n\n"
        "From the text above, please identify the two predicted root causes of the issue.\n\n"
        "Please print each of them in the form they appear in two separate lines."
        "I have a very rudimentary system, so if it is not in the exact form, it will crash."
    )
    message = router.client.complete(DB_LABEL_MODEL, [{"role": "user", "content": prompt}], meter=router.meter,
                                     component="judge", temperature=0.0, max_tokens=512)
    labels = [line.strip() for line in (message.content or "").strip().split("\n")]
    return {"fraction": sum(label in labels for label in gold) / len(gold), "gold": gold, "predicted_labels": labels}


def _rating_mean(values: list[float]) -> float | None:
    valid = [value for value in values if isinstance(value, (int, float)) and value >= 1]
    return sum(valid) / len(valid) if valid else None


def run_case(case: dict[str, Any], *, scenario: str, plugin: Any, marble: dict[str, Any], router: Router,
             run_root: Path, max_iterations: int) -> dict[str, Any]:
    import yaml

    case_dir = run_root / case["case_id"].replace("/", "_")
    if case_dir.exists():
        # On Windows/OneDrive, emptied directories can stay locked for a while; reuse them,
        # but never let a stale output or solution from an earlier attempt leak into this one.
        shutil.rmtree(case_dir, ignore_errors=True)
        stale = [p for p in (case_dir / "output.jsonl", case_dir / "marble" / "workspace" / "solution.py") if p.exists()]
        if stale:
            raise RuntimeError(f"could not clear previous attempt: {stale}")
    (case_dir / "marble" / "configs" / "coding_config").mkdir(parents=True, exist_ok=True)
    (case_dir / "marble" / "workspace").mkdir(parents=True, exist_ok=True)
    (case_dir / "evaluator").mkdir(parents=True, exist_ok=True)
    (case_dir / "logs").mkdir(parents=True, exist_ok=True)
    shutil.copy(MARBLE_ROOT / "marble" / "evaluator" / "evaluator_prompts.json", case_dir / "evaluator")
    config = json.loads(json.dumps(case["config"]))
    config["llm"] = ACTOR_MODEL
    config.setdefault("environment", {})["max_iterations"] = max_iterations
    if scenario == "coding":
        config["environment"]["workspace_dir"] = str(case_dir / "marble" / "workspace")
    config.setdefault("metrics", {})["evaluate_llm"] = JUDGE_ALIAS
    config["output"] = {"file_path": str(case_dir / "output.jsonl")}
    config_text = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    (case_dir / "config.yaml").write_text(config_text, encoding="utf-8")
    # Coding tools and the code-quality evaluator read this fixed relative path.
    (case_dir / "marble" / "configs" / "coding_config" / "coding_config.yaml").write_text(config_text, encoding="utf-8")

    task_text = str(config.get("task", {}).get("content", ""))
    _CTX.case_dir = case_dir
    meter_before = router.meter.snapshot()
    empty_before, truncated_before = router.empty_responses, router.truncated_tool_calls
    started = time.time()
    try:
        with _DB_LOCK if scenario == "database" else contextlib.nullcontext():
            engine = marble["Engine"](marble["Config"].load(str(case_dir / "config.yaml")))
            plugin.begin_task(case["case_id"], task_text[:10_000], f"MultiAgentBench {scenario}")
            memory_cls = plugin_memory_class(marble["BaseMemory"])
            for agent in engine.graph.get_all_agents():
                agent.memory = memory_cls(agent, plugin, task_text)
                if hasattr(plugin, "assign"):
                    plugin.assign(agent.agent_id, task_text[:4_000])
            engine.start()
        rows = [json.loads(line) for line in (case_dir / "output.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        summary = rows[-1] if rows else {}
        evaluator = marble["Evaluator"]({"evaluate_llm": JUDGE_ALIAS})
        # Task score.
        if scenario == "research":
            ratings = summary.get("task_evaluation") or {}
            task_rating = _rating_mean([ratings.get(k, -1) for k in ("innovation", "safety", "feasibility")])
            task_detail = ratings
        elif scenario == "database":
            task_detail = _database_task_score(router, summary.get("task_evaluation") or {})
            # Official TS is the matched fraction x 100; expressed here so that ts = rating x 20.
            task_rating = None if task_detail["fraction"] is None else 5 * task_detail["fraction"]
        else:
            solution = case_dir / "marble" / "workspace" / "solution.py"
            if solution.exists() and solution.read_text(encoding="utf-8", errors="replace").strip():
                task_detail = _code_quality(evaluator, task_text, solution)
            else:
                task_detail = {"instruction_following": 1, "executability": 1, "consistency": 1, "quality": 1,
                               "no_solution": True}
            task_rating = _rating_mean([task_detail.get(k, -1) for k in ("instruction_following", "executability", "consistency", "quality")])
        # Coordination score from the recorded iterations.
        profiles = "\n".join(f"Agent ID: {a.agent_id}, Profile: {a.profile}" for a in engine.graph.get_all_agents())
        comm_scores, plan_scores = [], []
        for iteration in summary.get("iterations", []):
            communications = iteration.get("communications") or []
            if communications:
                comm_scores.append(_graph_rating(evaluator, "Communication", {
                    "task": task_text, "communications": "\n".join(str(c) for c in communications)}))
            tasks = "\n".join(f"Agent {k}: Task: {v}" for k, v in (iteration.get("task_assignments") or {}).items())
            results = []
            for result in iteration.get("task_results") or []:
                if "agent_id" in result:
                    results.append(f"AgentID: {result['agent_id']}: Result: {result.get('result')}")
                else:
                    results.extend(f"Agent {k}: Result: {v}" for k, v in result.items())
            plan_scores.append(_graph_rating(evaluator, "Planning", {
                "summary": str(iteration.get("summary", "")), "agent_profiles": profiles,
                "agent_tasks": tasks, "results": "\n".join(results)}))
    finally:
        _CTX.case_dir = None
    comm, plan = _rating_mean(comm_scores), _rating_mean(plan_scores)
    cs_parts = [value for value in (comm, plan) if value is not None]
    ts = task_rating * 20 if task_rating is not None else None
    success = ts is not None and ts >= 60.0
    plugin.end_task(success=success, reward=(ts or 0.0) / 100.0)
    after = router.meter.snapshot()
    usage = {
        name: {k: after["components"].get(name, {}).get(k, 0) - meter_before["components"].get(name, {}).get(k, 0)
               for k in ("calls", "prompt_tokens", "completion_tokens")}
        for name in after["components"]
    }
    return {
        "case_id": case["case_id"], "ts": ts, "cs": sum(cs_parts) / len(cs_parts) * 20 if cs_parts else None,
        "communication": comm, "planning": plan, "task_detail": task_detail,
        "iterations": len(summary.get("iterations", [])), "router_usage": usage,
        "seconds": round(time.time() - started, 1), "empty_responses": router.empty_responses - empty_before,
        "truncated_tool_calls": router.truncated_tool_calls - truncated_before,
    }


def run_condition(scenario: str, method: str, *, client: LLMClient, marble: dict[str, Any], out: Path,
                  limit: int | None, max_iterations: int, rerun_errors: bool) -> None:
    """Run one condition's task sequence; memory persists across its tasks."""
    out_dir = out / scenario / method
    out_dir.mkdir(parents=True, exist_ok=True)
    router = Router(client, scenario)
    _CTX.router = router
    plugin = make_plugin(method, client, router.meter, out_dir / "pool", scenario)
    result_path = out_dir / "results.jsonl"
    done = set()
    if result_path.exists():
        for line in result_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if not (rerun_errors and "error" in row):
                    done.add(row["case_id"])
    cases = load_cases(scenario)
    cases = cases[:limit] if limit else cases
    tag = f"{scenario}/{method}"
    print(f"[{tag}] {len(done)} done, {len(cases) - len(done)} pending", flush=True)
    for case in cases:
        if case["case_id"] in done:
            continue
        # A crash (e.g. MARBLE's planner rejecting non-JSON output) is retried in place, so
        # the task keeps its position in the memory chain.
        for attempt in range(1, 4):
            try:
                row = run_case(case, scenario=scenario, plugin=plugin, marble=marble, router=router,
                               run_root=out_dir / "runs", max_iterations=max_iterations)
                row["attempts"] = attempt
                break
            except Exception as error:  # noqa: BLE001 - record and continue
                if hasattr(plugin, "abandon_task"):
                    plugin.abandon_task()
                row = {"case_id": case["case_id"], "error": f"{type(error).__name__}: {error}",
                       "traceback": traceback.format_exc()[-3000:], "attempts": attempt}
        with result_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        print(f"  [{tag}] {row['case_id']}: TS={row.get('ts')} CS={row.get('cs')} t={row.get('seconds')} "
              f"{row.get('error', '')[:150]}", flush=True)
        if hasattr(plugin, "pool_statistics"):
            (out_dir / "pool_statistics.json").write_text(json.dumps(plugin.pool_statistics(), indent=2), encoding="utf-8")
        (out_dir / "usage.json").write_text(json.dumps(router.meter.snapshot(), indent=2), encoding="utf-8")


def rescore_coding(method: str, *, client: LLMClient, marble: dict[str, Any], out: Path) -> int:
    """Re-score coding rows that got MARBLE's all-1 default although a solution exists.

    Only TS changes; the appended row (``rescored``) supersedes the original in reports.
    The memory chain itself already ran with the original TS as its feedback signal.
    """
    import yaml

    out_dir = out / "coding" / method
    path = out_dir / "results.jsonl"
    if not path.exists():
        return 0
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        if "error" not in row:
            latest[row["case_id"]] = row
    router = Router(client, "coding")
    _CTX.router = router
    fixed = 0
    for case_id, row in latest.items():
        detail = row.get("task_detail") or {}
        if row.get("rescored") or detail.get("no_solution") or any(detail.get(k) != 1 for k in CODE_KEYS):
            continue
        case_dir = out_dir / "runs" / case_id.replace("/", "_")
        solution = case_dir / "marble" / "workspace" / "solution.py"
        if not solution.exists() or not solution.read_text(encoding="utf-8", errors="replace").strip():
            continue
        config = yaml.safe_load((case_dir / "config.yaml").read_text(encoding="utf-8"))
        _CTX.case_dir = case_dir
        try:
            scores = _code_quality(marble["Evaluator"]({"evaluate_llm": JUDGE_ALIAS}),
                                   str(config.get("task", {}).get("content", "")), solution)
        finally:
            _CTX.case_dir = None
        rating = _rating_mean([scores.get(k, -1) for k in CODE_KEYS])
        new = {**row, "ts": rating * 20 if rating is not None else None, "task_detail": scores,
               "rescored": True, "ts_before_rescore": row.get("ts")}
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(new, ensure_ascii=False, default=str) + "\n")
        print(f"  [coding/{method}] {case_id}: rescored TS {row.get('ts')} -> {new['ts']}", flush=True)
        fixed += 1
    return fixed


def main(argv: list[str] | None = None) -> int:
    if not sys.flags.utf8_mode:
        # MARBLE opens files without an encoding; on Windows that writes cp1252 code which its
        # evaluator then fails to read as UTF-8. Re-launch in UTF-8 mode.
        import subprocess

        args = sys.argv[1:] if argv is None else argv
        return subprocess.call([sys.executable, "-X", "utf8", "-m", "sepm.bench.marble_mab", *args])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="+", required=True,
                        help="scenario:method pairs, e.g. research:sepm coding:no-memory")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--rerun-errors", action="store_true")
    parser.add_argument("--rescore", action="store_true", help="re-score all-1 coding rows, then exit")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "benchmark-results" / "multiagentbench")
    args = parser.parse_args(argv)
    conditions = []
    for item in args.conditions:
        scenario, method = item.split(":", 1)
        if scenario not in SCENARIO_SETTINGS or method not in METHODS:
            raise SystemExit(f"unknown condition {item!r}")
        conditions.append((scenario, method))

    client = LLMClient(env_file=str(PROJECT_ROOT / ".env"))
    out = args.out.resolve()
    embedder = CachedEmbedder(client, UsageMeter())
    if any(method in {"generative", "metagpt", "gmemory"} for _, method in conditions):
        from ..plugins._gmemory_compat import load_gmemory

        load_gmemory(PROJECT_ROOT / "external" / "GMemory")  # chdir-based import, before threads
    marble = install_marble(embedder)
    base_agent = marble["BaseAgent"]
    if not getattr(base_agent.act, "_sepm_wrapped", False):
        original_act = base_agent.act

        def act(self: Any, task: str) -> Any:
            self._sepm_current_task = task  # the agent's stated goal for goal alignment
            return original_act(self, task)

        act._sepm_wrapped = True
        base_agent.act = act
    if args.rescore:
        for scenario, method in conditions:
            if scenario == "coding":
                rescore_coding(method, client=client, marble=marble, out=out)
        return 0
    with ThreadPoolExecutor(max_workers=len(conditions)) as pool:
        futures = [pool.submit(run_condition, scenario, method, client=client, marble=marble, out=out,
                               limit=args.limit, max_iterations=args.max_iterations,
                               rerun_errors=args.rerun_errors) for scenario, method in conditions]
        for future in futures:
            future.result()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
