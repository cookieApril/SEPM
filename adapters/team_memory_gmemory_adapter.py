"""Run the official G-Memory task stack with Team Memory as ``meta_memory``.

The adapter intentionally reuses G-Memory's environments, MAS implementations, prompts,
and scheduling loop.  Only the memory object is replaced.  Actor calls use G-Memory's
native ``GPTChat`` path, while SOP curation uses the separately configured SOP endpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SUPPORTED_TASKS = ("alfworld", "sciworld", "pddl", "fever", "hotpotqa")
SUPPORTED_MEMORY_METHODS = ("no-memory", "gmemory", "team-memory")
TASK_METRICS = {
    "alfworld": "success_rate",
    "sciworld": "progress_rate",
    "pddl": "progress_rate",
    "fever": "exact_match",
    "hotpotqa": "exact_match",
}
EPISODE_CHECKPOINT_VERSION = 1
OFFICIAL_INDEX_KEY = "_team_memory_official_index"


def _parse_json_object(text: str) -> dict[str, Any]:
    """Parse one compact JSON object, accepting a surrounding Markdown fence."""
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise TypeError("SOP model response must be a JSON object")
    return value


def _sop_response_format_kwargs() -> dict[str, Any]:
    """Enable structured JSON output for local SOP models that require it."""
    if os.environ.get("TEAM_MEMORY_EVAL_SOP_RESPONSE_FORMAT") == "json_object":
        return {"response_format": {"type": "json_object"}}
    return {}


def _stable_identity_hash(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _patch_alfworld_textworld_eval_symbol() -> None:
    """Keep ALFWorld/TextWorld grammar evaluation compatible with modern Python."""
    import textworld.envs.pddl.textgen as textgen

    def _derive_with_explicit_eval_locals(self: Any, context: Any = None) -> list[Any]:
        context = context or self.context
        value = eval(self.expression, {}, context["variables"])
        return [textgen.TerminalSymbol(value)]

    textgen.EvalSymbol.derive = _derive_with_explicit_eval_locals


def _string_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a string or list")
    return [str(item) for item in value]


def _normalize_actor_action(text: str) -> str:
    """Normalize relay formatting without changing the action's semantics."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("actor model returned an empty completion")
    action = re.sub(r"^>\s*", "", lines[0])
    action = re.sub(r"^(?:Act\s*)?\d+\s*[:.)-]\s*", "", action, flags=re.IGNORECASE)
    action = re.sub(r"^>\s*", "", action)
    if not action.lower().startswith("think:"):
        action = action.rstrip().removesuffix(".").rstrip()
    if not action:
        raise RuntimeError("actor model returned an empty action after normalization")
    return action


def _official_case_id(task: dict[str, Any], index: int) -> str:
    env_kwargs = task.get("env_kwargs") or {}
    return str(task.get("id") or env_kwargs.get("gamefile") or task.get("gamefile") or index)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_episode_checkpoint(
    path: Path, identity: dict[str, Any]
) -> dict[str, Any]:
    if not path.exists():
        return {
            "version": EPISODE_CHECKPOINT_VERSION,
            "identity": identity,
            "cases": [],
            "actor_prompt_tokens": 0,
            "actor_completion_tokens": 0,
            "memory_stats": {},
            "complete": False,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != EPISODE_CHECKPOINT_VERSION:
        raise RuntimeError(f"unsupported episode checkpoint version: {path}")
    if payload.get("identity") != identity:
        raise RuntimeError(f"episode checkpoint identity mismatch: {path}")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise TypeError(f"episode checkpoint cases must be a list: {path}")
    indices = [case.get("official_task_index") for case in cases]
    if any(not isinstance(index, int) for index in indices) or len(indices) != len(set(indices)):
        raise RuntimeError(f"episode checkpoint has invalid or duplicate task indices: {path}")
    return payload


def _remaining_tasks(
    tasks: list[dict[str, Any]], completed_cases: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    completed = {case["official_task_index"] for case in completed_cases}
    available = {int(task[OFFICIAL_INDEX_KEY]) for task in tasks}
    unknown = completed - available
    if unknown:
        raise RuntimeError(f"checkpoint contains tasks outside the selected dataset: {sorted(unknown)}")
    return [task for task in tasks if int(task[OFFICIAL_INDEX_KEY]) not in completed]


def _prepare_scratch_dir(work_dir: Path) -> Path:
    """Use the large benchmark filesystem and discard only adapter-owned temp files."""
    scratch = (work_dir / "tmp").resolve()
    if scratch.parent != work_dir.resolve():
        raise RuntimeError(f"unsafe scratch path: {scratch}")
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    os.environ.update({"TMPDIR": str(scratch), "TMP": str(scratch), "TEMP": str(scratch)})
    tempfile.tempdir = None
    return scratch


def _close_environment(environment: Any) -> None:
    current = getattr(environment, "env", None)
    close = getattr(current, "close", None)
    if callable(close):
        close()


def _install_resource_safe_reset(environment: Any) -> None:
    original_reset = environment.reset

    def reset() -> Any:
        _close_environment(environment)
        return original_reset()

    environment.reset = reset


@dataclass
class StructuredRecorder:
    """Delegate official logging while retaining genuine per-task outcomes."""

    delegate: Any
    cases: list[dict[str, Any]] = field(default_factory=list)
    current_task_id: int | None = None
    current_task_config: dict[str, Any] | None = None
    base_seed: int = 0
    checkpoint_callback: Callable[[list[dict[str, Any]]], None] | None = None

    def log(self, message: str) -> None:
        self.delegate.log(message)

    def dataset_begin(self) -> None:
        self.delegate.dataset_begin()

    def dataset_end(self) -> None:
        self.delegate.dataset_end()

    def task_begin(self, task_id: int, task_config: dict[str, Any]) -> None:
        self.current_task_id = int(task_config.get(OFFICIAL_INDEX_KEY, task_id))
        self.current_task_config = task_config
        episode_seed = self.base_seed * 1_000_003 + self.current_task_id
        random.seed(episode_seed)
        try:
            import numpy as np

            np.random.seed(episode_seed % (2**32))
        except ImportError:
            pass
        self.delegate.task_begin(self.current_task_id, task_config)

    def task_end(self, reward: float, done: bool) -> None:
        self.delegate.task_end(reward, done)
        config = self.current_task_config or {}
        external_id = _official_case_id(config, self.current_task_id or 0)
        self.cases.append(
            {
                "case_id": str(external_id),
                "official_task_index": self.current_task_id,
                "reward": float(reward),
                "done": bool(done),
            }
        )
        if self.checkpoint_callback is not None:
            self.checkpoint_callback(self.cases)


def _select_cases(tasks: list[dict[str, Any]], case_id: str | None) -> list[dict[str, Any]]:
    if not case_id:
        return tasks
    selected: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        identifiers = {
            str(index),
            _official_case_id(task, index),
            str(task.get("id", "")),
            str(task.get("gamefile", "")),
            str(task.get("task", "")),
        }
        if case_id in identifiers:
            selected.append(task)
    if not selected:
        raise ValueError(f"official case id not found for this task: {case_id!r}")
    return selected


def _validate_official_data(gmemory_root: Path, task: str) -> None:
    """Fail before environment construction when the official data layout is incomplete."""
    required: list[Path]
    if task == "alfworld":
        required = [
            gmemory_root / "data/alfworld/alfworld_tasks_suffix.json",
            gmemory_root / "data/alfworld/json_2.1.1/valid_unseen",
            gmemory_root / "data/alfworld/logic/alfred.pddl",
            gmemory_root / "data/alfworld/logic/alfred.twl2",
        ]
    elif task == "sciworld":
        required = [gmemory_root / "data/sciworld/test.jsonl"]
    elif task == "pddl":
        required = [gmemory_root / "data/pddl/test.jsonl"]
    elif task == "fever":
        required = [gmemory_root / "data/fever/fever_dev.jsonl"]
    elif task == "hotpotqa":
        required = [gmemory_root / "data/hotpotqa/hotpot_dev_distractor_v1.json"]
    else:
        required = []
    missing = [path.as_posix() for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"official GMemory data for task {task!r} is incomplete; missing: {missing}"
        )


def _candidate_prompt(task_main: str, task_description: str, trajectory: str) -> str:
    return f"""Convert this completed multi-agent trajectory into a conservative candidate SOP.
Return exactly one JSON object with keys: title, description, applicability, exclusions,
atomic_steps, and rationale. atomic_steps must be a non-empty list of short action strings.
Do not claim causal validation and do not include hidden reasoning.

Task main:
{task_main}

Task description:
{task_description}

Observed action/observation trajectory:
{trajectory}
"""


def _build_team_memory_class():
    """Import both projects lazily, after runner-provided credentials are installed."""
    from mas.memory.common import MASMessage
    from mas.memory.mas_memory.memory_base import MASMemoryBase
    from openai import OpenAI

    from team_memory.ablation import config_from_environment
    from team_memory.adapter import AgentMemoryAdapter
    from team_memory.config import MemoryConfig
    from team_memory.models import (
        AgentProfile,
        BlackboardEntry,
        BlackboardKind,
        Evidence,
        EvidenceTier,
        ProcedureGraph,
        ProcedureStep,
        ProposalOperation,
        SOPCandidate,
        SOPMetadata,
        Workspace,
    )
    from team_memory.prompts import SOP_AGENT_SYSTEM_PROMPT
    from team_memory.service import TeamMemoryService

    @dataclass
    class TeamMemoryMASMemory(MASMemoryBase):
        database_path: Path = Path("team-memory.db")
        sop_model_name: str = ""
        sop_base_url: str = ""
        sop_api_key: str = ""
        task_family: str = ""
        experiment_id: str = ""
        seed: int = 0
        stats: dict[str, float] = field(default_factory=dict)

        def __post_init__(self) -> None:
            # Do not call G-Memory's persistence initializer: TeamMemoryService owns storage.
            memory_config = config_from_environment(
                MemoryConfig(database_path=self.database_path)
            )
            self.service = TeamMemoryService(config=memory_config)
            self.adapter = AgentMemoryAdapter(self.service)
            request_timeout = float(
                os.environ.get("TEAM_MEMORY_EVAL_REQUEST_TIMEOUT_SECONDS", "120")
            )
            if request_timeout <= 0:
                raise ValueError(
                    "TEAM_MEMORY_EVAL_REQUEST_TIMEOUT_SECONDS must be positive"
                )
            self.sop_client = OpenAI(
                base_url=self.sop_base_url,
                api_key=self.sop_api_key,
                timeout=request_timeout,
                max_retries=int(os.environ.get("TEAM_MEMORY_EVAL_MAX_RETRIES", "2")),
            )
            self.current_task_context = None
            self.workspace = None
            self.pending_agent = "solver"
            self.stats.update(
                {
                    "sop_model_calls": 0.0,
                    "sop_json_valid": 0.0,
                    "sop_candidates": 0.0,
                    "sop_candidates_passed": 0.0,
                    "sop_safety_rejections": 0.0,
                    "sop_transport_failures": 0.0,
                    "sop_prompt_tokens": 0.0,
                    "sop_completion_tokens": 0.0,
                }
            )

        def init_task_context(self, task_main: str, task_description: str | None = None):
            context = super().init_task_context(task_main, task_description)
            digest = hashlib.sha256(
                f"{self.experiment_id}|{self.task_family}|{self.seed}|{task_main}".encode()
            ).hexdigest()[:24]
            self.workspace = Workspace(workspace_id=f"gmemory-{digest}", main_goal=task_main)
            self.service.create_workspace(self.workspace)
            return context

        def add_agent_node(self, agent_message: Any, upstream_agent_ids: list[str]) -> str:
            node_id = super().add_agent_node(agent_message, upstream_agent_ids)
            self.pending_agent = agent_message.agent_name or "unknown-agent"
            self.service.register_agent(
                AgentProfile(
                    agent_id=self.pending_agent,
                    role=self.pending_agent,
                    current_task=self.current_task_context.task_main,
                )
            )
            return node_id

        def move_memory_state(self, action: str, observation: str, **kwargs: Any) -> None:
            super().move_memory_state(action, observation, **kwargs)
            assert self.workspace is not None
            evidence = Evidence(
                tier=EvidenceTier.TOOL_OBSERVATION,
                source=f"gmemory:{self.task_family}:environment",
                value=observation,
                verifier="official G-Memory environment",
            )
            self.adapter.on_agent_step(
                BlackboardEntry(
                    workspace_id=self.workspace.workspace_id,
                    agent_id=self.pending_agent,
                    kind=BlackboardKind.ACTION,
                    action=action,
                    observation=observation,
                    evidence=[evidence],
                )
            )

        def retrieve_memory(self, query_task: str, **kwargs: Any):
            ranked = self.service.retrieve_sops(query_task, kwargs.get("successful_topk", 1))
            trajectories: list[MASMessage] = []
            insights: list[str] = []
            for result in ranked:
                sop = result.sop
                steps = [step.instruction for step in sop.procedure.steps]
                trajectory = "\n".join(f"> {step}" for step in steps)
                message = MASMessage(
                    task_main=sop.metadata.title,
                    task_description=sop.metadata.description or sop.metadata.title,
                    task_trajectory=trajectory,
                    label=True,
                )
                message.add_extra_field("key_steps", "\n".join(steps))
                trajectories.append(message)
                insights.extend(steps)
            return trajectories, [], insights

        def add_memory(self, mas_message: Any) -> None:
            # A model may propose a candidate, but it never supplies causal evidence.
            # Therefore a post-hoc successful trace is not promoted by this adapter.
            if mas_message.label is not True:
                return
            self.stats["sop_model_calls"] += 1.0
            try:
                response = self.sop_client.chat.completions.create(
                    model=self.sop_model_name,
                    messages=[
                        {"role": "system", "content": SOP_AGENT_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": _candidate_prompt(
                                mas_message.task_main,
                                mas_message.task_description or "",
                                mas_message.task_trajectory or "",
                            ),
                        },
                    ],
                    temperature=0.0,
                    max_tokens=1200,
                    **_sop_response_format_kwargs(),
                )
            except Exception as error:
                # SOP curation is post-hoc. A provider timeout must not erase a
                # completed environment episode or prevent its checkpoint from
                # being written. Record the transport failure and reject only
                # this candidate; a later explicit failed-cell run may retry it.
                self.stats["sop_transport_failures"] += 1.0
                print(f"SOP candidate transport rejected: {error}", file=sys.stderr)
                return
            usage = getattr(response, "usage", None)
            self.stats["sop_prompt_tokens"] += float(getattr(usage, "prompt_tokens", 0) or 0)
            self.stats["sop_completion_tokens"] += float(
                getattr(usage, "completion_tokens", 0) or 0
            )
            try:
                raw = _parse_json_object(response.choices[0].message.content or "")
                raw_steps = raw["atomic_steps"]
                if not isinstance(raw_steps, list) or not raw_steps:
                    raise ValueError("atomic_steps must be a non-empty list")
                steps = [
                    ProcedureStep(step_id=f"step-{index}", instruction=str(step))
                    for index, step in enumerate(raw_steps, 1)
                ]
                applicability = _string_list(raw.get("applicability"), "applicability")
                exclusions = _string_list(raw.get("exclusions"), "exclusions")
                candidate = SOPCandidate(
                    operation=ProposalOperation.CREATE,
                    procedure=ProcedureGraph(steps=steps),
                    metadata=SOPMetadata(
                        title=str(raw["title"]),
                        description=str(raw.get("description", raw.get("rationale", ""))),
                        task_family=self.task_family,
                        applicability=applicability,
                        exclusions=exclusions,
                        tags=["gmemory-adapter", self.task_family],
                    ),
                    source_workspace_ids=[self.workspace.workspace_id],
                    state_verified=True,
                    causal_confidence=0.0,
                )
                self.stats["sop_json_valid"] += 1.0
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                print(f"SOP candidate parse rejected: {error}", file=sys.stderr)
                return
            stored, safety = self.service.propose_sop(candidate)
            self.stats["sop_candidates"] += 1.0
            if not safety.allowed:
                self.stats["sop_safety_rejections"] += 1.0
                return
            validated, _ = self.service.validate_candidate(stored.candidate_id)
            if validated.status.value in {"validated", "promoted"}:
                self.stats["sop_candidates_passed"] += 1.0

        def backward(self, reward: Any, **kwargs: Any) -> None:
            return None

    return TeamMemoryMASMemory


def run(args: argparse.Namespace) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, Any]]:
    project_root = Path(__file__).resolve().parents[1]
    gmemory_root = Path.cwd().resolve()
    if not (gmemory_root / "tasks" / "run.py").is_file():
        raise FileNotFoundError(
            f"adapter must run from the GMemory repository; missing {gmemory_root / 'tasks/run.py'}"
        )
    sys.path.insert(0, str(project_root / "src"))
    sys.path.insert(0, str(gmemory_root))
    sys.path.insert(0, str(gmemory_root / "tasks"))

    if args.task not in SUPPORTED_TASKS:
        raise ValueError(f"unsupported GMemory task: {args.task}")
    _validate_official_data(gmemory_root, args.task)

    actor_base_url = os.environ["TEAM_MEMORY_EVAL_ACTOR_BASE_URL"]
    actor_api_key = os.environ["TEAM_MEMORY_EVAL_ACTOR_API_KEY"]
    sop_base_url = os.environ["TEAM_MEMORY_EVAL_SOP_BASE_URL"]
    sop_api_key = os.environ["TEAM_MEMORY_EVAL_SOP_API_KEY"]
    # G-Memory reads these at module import time.
    os.environ["OPENAI_API_BASE"] = actor_base_url
    os.environ["OPENAI_BASE_URL"] = actor_base_url
    os.environ["OPENAI_API_KEY"] = actor_api_key

    output = Path(args.output).resolve()
    work_dir = output.parent.parent / "gmemory-work" / output.stem
    work_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = _prepare_scratch_dir(work_dir)

    import run as official_run
    from mas.module_map import module_map

    TeamMemoryMASMemory = _build_team_memory_class()
    memory_method = args.memory_method or os.environ.get("TEAM_MEMORY_EVAL_METHOD")
    if memory_method not in SUPPORTED_MEMORY_METHODS:
        raise ValueError(f"unsupported memory method: {memory_method!r}")

    class RelayActorChat:
        """GMemory LLMCallable with bounded, relay-compatible request arguments."""

        def __init__(self, model_name: str) -> None:
            from openai import OpenAI

            self.model_name = model_name
            self.default_max_tokens = int(
                os.environ.get("TEAM_MEMORY_EVAL_ACTOR_MAX_TOKENS", "4096")
            )
            self.max_retries = int(os.environ.get("TEAM_MEMORY_EVAL_MAX_RETRIES", "2"))
            if self.default_max_tokens < 1:
                raise ValueError("TEAM_MEMORY_EVAL_ACTOR_MAX_TOKENS must be positive")
            if self.max_retries < 0:
                raise ValueError("TEAM_MEMORY_EVAL_MAX_RETRIES cannot be negative")
            request_timeout = float(
                os.environ.get("TEAM_MEMORY_EVAL_REQUEST_TIMEOUT_SECONDS", "120")
            )
            if request_timeout <= 0:
                raise ValueError(
                    "TEAM_MEMORY_EVAL_REQUEST_TIMEOUT_SECONDS must be positive"
                )
            self.client = OpenAI(
                base_url=actor_base_url,
                api_key=actor_api_key,
                timeout=request_timeout,
                max_retries=self.max_retries,
            )
            self.prompt_tokens = 0
            self.completion_tokens = 0

        def __call__(
            self,
            messages: list[Any],
            temperature: float | None = None,
            max_tokens: int | None = None,
            stop_strs: list[str] | None = None,
            num_comps: int | None = None,
        ) -> str:
            # The configured relay times out when the legacy GMemory client sends
            # stop=["\n"]. Official env.process_action already selects the first line.
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": item.role, "content": item.content} for item in messages],
                temperature=0.0 if temperature is None else temperature,
                max_tokens=self.default_max_tokens if max_tokens is None else max_tokens,
                n=1 if num_comps is None else num_comps,
            )
            usage = getattr(response, "usage", None)
            self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
            content = response.choices[0].message.content or ""
            action = _normalize_actor_action(content)
            if not action:
                choice = response.choices[0]
                print(
                    "Relay actor returned an empty action: "
                    f"response_id={getattr(response, 'id', None)!r}, "
                    f"finish_reason={getattr(choice, 'finish_reason', None)!r}",
                    flush=True,
                )
            return action

    random.seed(args.seed)
    try:
        import numpy as np

        np.random.seed(args.seed)
    except ImportError:
        pass

    official_run.WORKING_DIR = str(work_dir)
    configured_steps = int(official_run.CONFIG[args.task].get("max_steps", 30))
    if args.task == "alfworld":
        _patch_alfworld_textworld_eval_symbol()
    manager = official_run.build_task(args.task, args.mas, "team-memory", configured_steps)
    # ALFWorld creates a new underlying environment on reset and must close the
    # previous one first. ScienceWorld reuses one Py4J gateway across load/reset;
    # closing it here makes every subsequent reset fail with "Gateway is not
    # connected." Other environments do not need this wrapper either.
    if args.task == "alfworld":
        _install_resource_safe_reset(manager.env)
    case_id = os.environ.get("TEAM_MEMORY_EVAL_CASE_ID") or None
    indexed_tasks: list[dict[str, Any]] = []
    for index, task in enumerate(manager.tasks):
        copied = copy.deepcopy(task)
        copied[OFFICIAL_INDEX_KEY] = index
        indexed_tasks.append(copied)
    selected_tasks = _select_cases(indexed_tasks, case_id)
    checkpoint_path = work_dir / "episode-checkpoint.json"
    checkpoint_identity = {
        "task": args.task,
        "mas": args.mas,
        "memory_method": memory_method,
        "actor_model": args.actor_model,
        "sop_model": args.sop_model,
        "seed": args.seed,
        "case_id": case_id,
        "selected_cases": [
            _official_case_id(task, int(task[OFFICIAL_INDEX_KEY])) for task in selected_tasks
        ],
        "episode_seed_policy": "base_seed*1000003+official_task_index",
    }
    checkpoint = _load_episode_checkpoint(checkpoint_path, checkpoint_identity)
    manager.tasks = _remaining_tasks(selected_tasks, checkpoint["cases"])
    recorder = StructuredRecorder(
        manager.recorder,
        cases=list(checkpoint["cases"]),
        base_seed=args.seed,
    )
    manager.recorder = recorder

    database_value = os.environ.get("TEAM_MEMORY_DB")
    database_path = (
        Path(database_value).resolve() if database_value else output.with_suffix(".team-memory.db")
    )
    actor_chat = RelayActorChat(args.actor_model)
    actor_chat.prompt_tokens = int(checkpoint.get("actor_prompt_tokens", 0))
    actor_chat.completion_tokens = int(checkpoint.get("actor_completion_tokens", 0))
    if memory_method == "team-memory":
        memory = TeamMemoryMASMemory(
            namespace="team-memory",
            global_config={},
            llm_model=None,
            embedding_func=None,
            database_path=database_path,
            sop_model_name=args.sop_model,
            sop_base_url=sop_base_url,
            sop_api_key=sop_api_key,
            task_family=args.task,
            experiment_id=output.stem,
            seed=args.seed,
        )
    else:
        from mas.utils import EmbeddingFunc

        official_memory_name = "empty" if memory_method == "no-memory" else "g-memory"
        _, memory_type = module_map("io", official_memory_name)
        embedding = EmbeddingFunc(official_run.CONFIG["embedding_model"]) if memory_method == "gmemory" else None
        memory = memory_type(
            namespace=official_memory_name,
            global_config={"working_dir": str(work_dir), "hop": 1},
            llm_model=actor_chat,
            embedding_func=embedding,
        )
    memory_stats = getattr(memory, "stats", None)
    if isinstance(memory_stats, dict):
        memory_stats.update(
            {key: float(value) for key, value in checkpoint.get("memory_stats", {}).items()}
        )

    def save_episode_checkpoint(
        cases: list[dict[str, Any]], *, complete: bool = False
    ) -> None:
        stats = getattr(memory, "stats", {})
        _atomic_write_json(
            checkpoint_path,
            {
                "version": EPISODE_CHECKPOINT_VERSION,
                "identity": checkpoint_identity,
                "cases": cases,
                "actor_prompt_tokens": actor_chat.prompt_tokens,
                "actor_completion_tokens": actor_chat.completion_tokens,
                "memory_stats": dict(stats) if isinstance(stats, dict) else {},
                "complete": complete,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )

    recorder.checkpoint_callback = save_episode_checkpoint
    reasoning_type, _ = module_map("io", None)
    reasoning = reasoning_type(llm_model=actor_chat)
    manager.mas.add_observer(recorder)
    manager.mas.build_system(reasoning, memory, manager.env, manager.mas_config)
    recorder.log(
        f"Episode checkpoint resume: completed={len(recorder.cases)}, "
        f"remaining={len(manager.tasks)}, path={checkpoint_path}"
    )
    try:
        official_run.run_task(manager)
        save_episode_checkpoint(recorder.cases, complete=True)
    finally:
        _close_environment(manager.env)
        shutil.rmtree(scratch_dir, ignore_errors=True)

    rewards = [float(case["reward"]) for case in recorder.cases]
    dones = [1.0 if case["done"] else 0.0 for case in recorder.cases]
    primary_name = TASK_METRICS[args.task]
    primary_score = _mean(dones) if args.task in {"alfworld", "fever", "hotpotqa"} else _mean(rewards)
    memory_stats = getattr(memory, "stats", {})
    calls = float(memory_stats.get("sop_model_calls", 0.0))
    candidates = float(memory_stats.get("sop_candidates", 0.0))
    metrics = {
        primary_name: primary_score,
        "primary_score": primary_score,
        "case_count": float(len(recorder.cases)),
        "mean_reward": _mean(rewards),
        "success_rate": _mean(dones),
        "actor_prompt_tokens": float(actor_chat.prompt_tokens),
        "actor_completion_tokens": float(actor_chat.completion_tokens),
        **memory_stats,
        "sop_json_valid_rate": float(memory_stats.get("sop_json_valid", 0.0)) / calls if calls else 0.0,
        "sop_candidate_pass_rate": (
            float(memory_stats.get("sop_candidates_passed", 0.0)) / candidates if candidates else 0.0
        ),
    }
    metadata = {
        "adapter": "adapters/team_memory_gmemory_adapter.py",
        "memory_method": memory_method,
        "host_execution_mode": f"{args.task}:{args.mas}:gmemory-official-mas",
        "host_adapter_identity": _stable_identity_hash(
            args.task, args.mas, "external/GMemory/tasks/run.py components"
        ),
        "host_topology_hash": _stable_identity_hash(
            args.task, args.mas, "external/GMemory/tasks/mas_workflow"
        ),
        "agent_roles": [args.mas],
        "message_routing_mode": args.mas,
        "memory_execution_mode": f"{args.task}:{memory_method}:gmemory-official-memory",
        "memory_adapter_identity": _stable_identity_hash(
            args.task, memory_method, "external/GMemory/tasks/memory"
        ),
        "memory_enabled": memory_method != "no-memory",
        "result_schema_version": "e1-runtime-identity-v1",
        "official_entrypoint": "external/GMemory/tasks/run.py components",
        "official_repository": str(gmemory_root),
        "team_memory_db": str(database_path),
        "actor_model": args.actor_model,
        "sop_model": args.sop_model,
        "actor_stop_policy": "relay-incompatible newline stop omitted; env selects first line",
        "actor_max_tokens": actor_chat.default_max_tokens,
        "episode_checkpoint": str(checkpoint_path),
        "resumed_case_count": len(checkpoint["cases"]),
        "scratch_policy": "adapter-owned work_dir/tmp; old env closed before reset",
        "causal_policy": "post-hoc candidates remain unpromoted without external causal evidence",
    }
    return metrics, recorder.cases, metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--mas", required=True, choices=("autogen", "dylan", "macnet"))
    parser.add_argument("--actor-model", required=True)
    parser.add_argument("--sop-model", required=True)
    parser.add_argument("--memory-method", choices=SUPPORTED_MEMORY_METHODS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from team_memory.evaluation_adapter import EvaluationContext

    context = EvaluationContext.from_environment()
    expected = {
        "task": args.task,
        "mas": args.mas,
        "actor_model": args.actor_model,
        "sop_model": args.sop_model,
        "seed": args.seed,
        "output": args.output.resolve(),
        "memory_method": args.memory_method or context.memory_method,
    }
    actual = {
        "task": context.task,
        "mas": context.mas_framework,
        "actor_model": context.actor_model,
        "sop_model": context.sop_model,
        "seed": context.seed,
        "output": context.output_path.resolve(),
        "memory_method": context.memory_method,
    }
    if expected != actual:
        raise RuntimeError(f"CLI/environment experiment identity mismatch: {expected} != {actual}")
    metrics, cases, metadata = run(args)
    context.write_result(metrics, cases=cases, metadata=metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
