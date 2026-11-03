"""Run the official G-Memory task stack with SEPM as ``meta_memory``.

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
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SUPPORTED_TASKS = ("alfworld", "officebench", "sciworld", "pddl", "fever", "hotpotqa")
SUPPORTED_MEMORY_METHODS = ("no-memory", "generative", "metagpt", "gmemory", "sepm")
# Official G-Memory module_map names of the baseline memories.
OFFICIAL_MEMORY_NAMES = {"no-memory": "empty", "generative": "generative", "metagpt": "metagpt", "gmemory": "g-memory"}
TASK_METRICS = {
    "alfworld": "success_rate",
    "officebench": "pass_rate",
    "sciworld": "progress_rate",
    "pddl": "progress_rate",
    "fever": "exact_match",
    "hotpotqa": "exact_match",
}
EPISODE_CHECKPOINT_VERSION = 1
OFFICIAL_INDEX_KEY = "_sepm_official_index"


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
    if os.environ.get("SEPM_EVAL_SOP_RESPONSE_FORMAT") == "json_object":
        return {"response_format": {"type": "json_object"}}
    return {}


def _stable_identity_hash(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _patch_alfworld_textworld_eval_symbol() -> None:
    """Keep ALFWorld/TextWorld grammar evaluation compatible with modern Python."""
    from textworld.envs.pddl import textgen

    def _derive_with_explicit_eval_locals(self: Any, context: Any = None) -> list[Any]:
        context = context or self.context
        value = eval(self.expression, {}, context["variables"])
        return [textgen.TerminalSymbol(value)]

    textgen.EvalSymbol.derive = _derive_with_explicit_eval_locals


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
    # Returns cumulative memory/actor counters; per-task deltas are stored with each case.
    counters_provider: Callable[[], dict[str, float]] | None = None
    _task_started: float = 0.0
    _counters_at_start: dict[str, float] = field(default_factory=dict)

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
        self._task_started = time.time()
        self._counters_at_start = dict(self.counters_provider()) if self.counters_provider else {}

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
                "task_type": config.get("task_type"),
                "seconds": round(time.time() - self._task_started, 1),
                "counters": {
                    key: value - self._counters_at_start.get(key, 0.0)
                    for key, value in (self.counters_provider() if self.counters_provider else {}).items()
                    if value - self._counters_at_start.get(key, 0.0)
                },
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
    if task == "officebench":
        required = [gmemory_root.parent / "OfficeBench" / "tasks", gmemory_root.parent / "OfficeBench" / "apps"]
    elif task == "alfworld":
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


def _pool_statistics(memory: Any) -> dict[str, Any]:
    """Procedure-pool statistics (paper Table 9) read from the SEPM store."""
    from sepm.models import CandidateStatus

    store = memory.service.store
    by_status = {status.value: store.list_candidates(status, limit=1, offset=0)[1] for status in CandidateStatus}
    active, active_total = store.list_active_sops(limit=10_000, offset=0)
    return {
        "candidates_by_status": by_status,
        "active_procedures": active_total,
        "specialized_variants": sum(1 for sop in active if sop.metadata.variant_of),
        "warnings_on_active": sum(len(sop.metadata.warnings) for sop in active),
        "avg_versions_per_active": sum(sop.version for sop in active) / len(active) if active else 0.0,
    }


def _filter_task_types(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Leave-one-type-out support: SEPM_TASK_TYPES keeps, SEPM_EXCLUDE_TASK_TYPES drops task types."""
    include = {item for item in os.environ.get("SEPM_TASK_TYPES", "").split(",") if item}
    exclude = {item for item in os.environ.get("SEPM_EXCLUDE_TASK_TYPES", "").split(",") if item}
    return [
        task for task in tasks
        if (not include or task.get("task_type") in include) and task.get("task_type") not in exclude
    ]


def _build_officebench_manager(official_run: Any, mas_type: str, gmemory_root: Path, max_steps: int, work_dir: Path) -> Any:
    """TaskManager for OfficeBench using G-Memory's own MAS workflows and recorder."""
    from envs import BaseRecorder
    from mas_workflow import get_mas

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from officebench_env import OfficeBenchEnv, officebench_tasks

    office_root = gmemory_root.parent / "OfficeBench"
    env = OfficeBenchEnv(office_root, docker_name=os.environ.get("OFFICEBENCH_IMAGE", "officebench"),
                         container_name=os.environ.get("OFFICEBENCH_CONTAINER", f"officebench-{os.getpid()}"),
                         max_trials=max_steps, output_root=work_dir / "officebench-outputs")
    return official_run.TaskManager(
        task_name="officebench", mas_type=mas_type, memory_type="sepm", tasks=officebench_tasks(office_root),
        env=env, recorder=BaseRecorder(working_dir=str(work_dir), namespace="total_task"),
        mas=get_mas(mas_type), mas_config=dict(official_run.CONFIG.get(mas_type, {})),
    )


def _run_officebench_tasks(manager: Any) -> None:
    """G-Memory's run_task loop with OfficeBench's instruction instead of a G-Memory dataset prompt."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from officebench_env import OFFICEBENCH_INSTRUCTION

    manager.recorder.dataset_begin()
    try:
        for task_id, task_config in enumerate(manager.tasks):
            manager.recorder.task_begin(task_id, task_config)
            task_main, task_description = manager.mas.env.set_env(task_config)
            task_config.update(task_main=task_main, task_description=task_description, few_shots=[])
            for agent in manager.mas.agents_team.values():
                agent.add_task_instruction(OFFICEBENCH_INSTRUCTION)
            reward, done = manager.mas.schedule(task_config)
            manager.recorder.task_end(reward, done)
    finally:
        manager.mas.env.close()
    manager.recorder.dataset_end()


def _build_sepm_class():
    """Import both projects lazily, after runner-provided credentials are installed."""
    from mas.memory.common import MASMessage
    from mas.memory.mas_memory.memory_base import MASMemoryBase
    from openai import OpenAI

    from sepm.ablation import config_from_environment
    from sepm.config import MemoryConfig
    from sepm.domains import domain_for
    from sepm.episode import ProceduralEpisode, render_procedure
    from sepm.service import SEPMService

    class OpenAISOPModel:
        """SOP endpoint returning one parsed JSON object; usage is added to adapter stats."""

        def __init__(self, client: Any, model_name: str, stats: dict[str, float]) -> None:
            self.client = client
            self.model_name = model_name
            self.stats = stats

        def complete_json(self, system: str, user: str) -> dict[str, Any]:
            self.stats["sop_model_calls"] += 1.0
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    max_tokens=1200,
                    **_sop_response_format_kwargs(),
                )
            except Exception:
                # Curation is post-hoc; the episode result stands and only this update is lost.
                self.stats["sop_transport_failures"] += 1.0
                raise
            usage = getattr(response, "usage", None)
            self.stats["sop_prompt_tokens"] += float(getattr(usage, "prompt_tokens", 0) or 0)
            self.stats["sop_completion_tokens"] += float(getattr(usage, "completion_tokens", 0) or 0)
            parsed = _parse_json_object(response.choices[0].message.content or "")
            self.stats["sop_json_valid"] += 1.0
            return parsed

    @dataclass
    class SEPMMASMemory(MASMemoryBase):
        database_path: Path = Path("sepm.db")
        sop_model_name: str = ""
        sop_base_url: str = ""
        sop_api_key: str = ""
        task_family: str = ""
        experiment_id: str = ""
        seed: int = 0
        max_steps: int = 30
        stats: dict[str, float] = field(default_factory=dict)

        def __post_init__(self) -> None:
            # Do not call G-Memory's persistence initializer: SEPMService owns storage.
            memory_config = config_from_environment(
                MemoryConfig(database_path=self.database_path)
            )
            self.service = SEPMService(config=memory_config)
            self.domain = domain_for(self.task_family)
            request_timeout = float(
                os.environ.get("SEPM_EVAL_REQUEST_TIMEOUT_SECONDS", "120")
            )
            if request_timeout <= 0:
                raise ValueError(
                    "SEPM_EVAL_REQUEST_TIMEOUT_SECONDS must be positive"
                )
            for key in (
                "sop_model_calls",
                "sop_json_valid",
                "sop_candidates",
                "sop_candidates_passed",
                "sop_safety_rejections",
                "sop_transport_failures",
                "sop_prompt_tokens",
                "sop_completion_tokens",
            ):
                self.stats.setdefault(key, 0.0)
            self.sop_model = OpenAISOPModel(
                OpenAI(
                    base_url=self.sop_base_url,
                    api_key=self.sop_api_key,
                    timeout=request_timeout,
                    max_retries=int(os.environ.get("SEPM_EVAL_MAX_RETRIES", "2")),
                ),
                self.sop_model_name,
                self.stats,
            )
            self.current_task_context = None
            self.episode: ProceduralEpisode | None = None
            self.pending_agent = "solver"

        def init_task_context(self, task_main: str, task_description: str | None = None):
            context = super().init_task_context(task_main, task_description)
            digest = hashlib.sha256(
                f"{self.experiment_id}|{self.task_family}|{self.seed}|{task_main}".encode()
            ).hexdigest()[:24]
            # The suffix keeps repeated task texts and retried episodes in separate workspaces.
            self.episode = ProceduralEpisode(
                self.service,
                self.domain,
                workspace_id=f"gmemory-{digest}-{uuid.uuid4().hex[:8]}",
                task_goal=task_main,
                sop_model=self.sop_model,
                stats=self.stats,
                max_steps=self.max_steps,
            )
            return context

        def _require_episode(self) -> ProceduralEpisode:
            if self.episode is None:
                raise RuntimeError("init_task_context must run before SEPM memory hooks")
            return self.episode

        def add_agent_node(self, agent_message: Any, upstream_agent_ids: list[str]) -> str:
            node_id = super().add_agent_node(agent_message, upstream_agent_ids)
            self.pending_agent = agent_message.agent_name or "unknown-agent"
            self._require_episode().register_agent(self.pending_agent)
            return node_id

        def move_memory_state(self, action: str, observation: str, **kwargs: Any) -> None:
            super().move_memory_state(action, observation, **kwargs)
            self._require_episode().record_step(self.pending_agent, str(action), str(observation))

        def take_alignment_notices(self) -> list[str]:
            """Corrections for the next actor call; empty outside an episode."""
            return self.episode.take_notices() if self.episode is not None else []

        def retrieve_memory(self, query_task: str, **kwargs: Any):
            context = self._require_episode().retrieve(
                query_task, int(kwargs.get("successful_topk") or 5)
            )
            injected = [(sop, "") for sop in context.active]
            if context.trial is not None:
                injected.append(
                    (context.trial, "Candidate procedure under validation; verify each step.\n")
                )
            trajectories: list[MASMessage] = []
            for sop, prefix in injected:
                message = MASMessage(
                    task_main=sop.metadata.title,
                    task_description=sop.metadata.description or sop.metadata.title,
                    task_trajectory=prefix + render_procedure(sop, self.service.config.explicit_task_graph),
                    label=True,
                )
                message.add_extra_field(
                    "key_steps", "\n".join(step.instruction for step in sop.procedure.steps)
                )
                trajectories.append(message)
            insights = [
                warning.text
                + (f" (when: {warning.trigger_condition})" if warning.trigger_condition else "")
                for sop in context.active
                for warning in sop.metadata.warnings
            ]
            return trajectories, [], insights

        def add_memory(self, mas_message: Any) -> None:
            # MASMemoryBase.save_task_context calls this once per episode with the official label.
            success = mas_message.label is True
            self._require_episode().finish(success=success, reward=1.0 if success else 0.0)

        def backward(self, reward: Any, **kwargs: Any) -> None:
            return None

    return SEPMMASMemory


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

    actor_base_url = os.environ["SEPM_EVAL_ACTOR_BASE_URL"]
    actor_api_key = os.environ["SEPM_EVAL_ACTOR_API_KEY"]
    sop_base_url = os.environ["SEPM_EVAL_SOP_BASE_URL"]
    sop_api_key = os.environ["SEPM_EVAL_SOP_API_KEY"]
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

    SEPMMASMemory = _build_sepm_class()
    memory_method = args.memory_method or os.environ.get("SEPM_EVAL_METHOD")
    if memory_method not in SUPPORTED_MEMORY_METHODS:
        raise ValueError(f"unsupported memory method: {memory_method!r}")

    class RelayActorChat:
        """GMemory LLMCallable with bounded, relay-compatible request arguments."""

        def __init__(self, model_name: str) -> None:
            from openai import OpenAI

            self.model_name = model_name
            self.default_max_tokens = int(
                os.environ.get("SEPM_EVAL_ACTOR_MAX_TOKENS", "4096")
            )
            self.max_retries = int(os.environ.get("SEPM_EVAL_MAX_RETRIES", "2"))
            if self.default_max_tokens < 1:
                raise ValueError("SEPM_EVAL_ACTOR_MAX_TOKENS must be positive")
            if self.max_retries < 0:
                raise ValueError("SEPM_EVAL_MAX_RETRIES cannot be negative")
            request_timeout = float(
                os.environ.get("SEPM_EVAL_REQUEST_TIMEOUT_SECONDS", "120")
            )
            if request_timeout <= 0:
                raise ValueError(
                    "SEPM_EVAL_REQUEST_TIMEOUT_SECONDS must be positive"
                )
            self.client = OpenAI(
                base_url=actor_base_url,
                api_key=actor_api_key,
                timeout=request_timeout,
                max_retries=self.max_retries,
            )
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.calls = 0
            # Supplies SEPM realignment corrections for the next actor call, if any.
            self.notice_provider: Callable[[], list[str]] | None = None

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
            payload = [{"role": item.role, "content": item.content} for item in messages]
            notices = self.notice_provider() if self.notice_provider is not None else []
            if notices and payload:
                # Propagate the manager's correction to the acting agent (Algorithm 1, line 8).
                payload[-1]["content"] += "\n\nSEPM alignment notice:\n" + "\n".join(
                    f"- {notice}" for notice in notices
                )
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=payload,
                temperature=0.0 if temperature is None else temperature,
                max_tokens=self.default_max_tokens if max_tokens is None else max_tokens,
                n=1 if num_comps is None else num_comps,
            )
            usage = getattr(response, "usage", None)
            self.calls += 1
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
    if args.task == "officebench":
        configured_steps = int(os.environ.get("SEPM_OFFICEBENCH_MAX_STEPS", "20"))
        manager = _build_officebench_manager(official_run, args.mas, gmemory_root, configured_steps, work_dir)
    else:
        configured_steps = int(official_run.CONFIG[args.task].get("max_steps", 30))
        if args.task == "alfworld":
            _patch_alfworld_textworld_eval_symbol()
        manager = official_run.build_task(args.task, args.mas, "sepm", configured_steps)
    if os.environ.get("SEPM_AGENT_COUNT"):
        # Team-size analysis (E10): number of task agents in the host.
        # Only DyLAN exposes a team size (node_num); G-Memory's AutoGen has fixed roles.
        manager.mas_config["node_num"] = int(os.environ["SEPM_AGENT_COUNT"])
    # ALFWorld creates a new underlying environment on reset and must close the
    # previous one first. ScienceWorld reuses one Py4J gateway across load/reset;
    # closing it here makes every subsequent reset fail with "Gateway is not
    # connected." Other environments do not need this wrapper either.
    if args.task == "alfworld":
        _install_resource_safe_reset(manager.env)
    case_id = os.environ.get("SEPM_EVAL_CASE_ID") or None
    indexed_tasks: list[dict[str, Any]] = []
    for index, task in enumerate(manager.tasks):
        copied = copy.deepcopy(task)
        copied[OFFICIAL_INDEX_KEY] = index
        indexed_tasks.append(copied)
    selected_tasks = _filter_task_types(_select_cases(indexed_tasks, case_id))
    if args.seed > 0 and not case_id:
        # Seeds 1 and 2 use independent permutations of the adapter's task order (App. C.4).
        random.Random(args.seed).shuffle(selected_tasks)
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

    database_value = os.environ.get("SEPM_DB")
    database_path = (
        Path(database_value).resolve() if database_value else output.with_suffix(".sepm.db")
    )
    actor_chat = RelayActorChat(args.actor_model)
    actor_chat.prompt_tokens = int(checkpoint.get("actor_prompt_tokens", 0))
    actor_chat.completion_tokens = int(checkpoint.get("actor_completion_tokens", 0))
    if memory_method == "sepm":
        memory = SEPMMASMemory(
            namespace="sepm",
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
            max_steps=configured_steps,
        )
        actor_chat.notice_provider = memory.take_alignment_notices
    else:
        from mas.utils import EmbeddingFunc

        official_memory_name = OFFICIAL_MEMORY_NAMES[memory_method]
        _, memory_type = module_map("io", official_memory_name)
        embedding = EmbeddingFunc(official_run.CONFIG["embedding_model"]) if memory_method != "no-memory" else None
        # A shared directory lets a pool be built in one run and reused (frozen) in another.
        memory_dir = Path(os.environ.get("SEPM_GMEMORY_WORKDIR") or work_dir)
        memory = memory_type(
            namespace=official_memory_name,
            global_config={"working_dir": str(memory_dir), "hop": 1},
            llm_model=actor_chat,
            embedding_func=embedding,
        )
        if os.environ.get("SEPM_FREEZE_POOL") == "1":
            memory.add_memory = lambda *_args, **_kwargs: None
            memory.backward = lambda *_args, **_kwargs: None
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
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    recorder.checkpoint_callback = save_episode_checkpoint

    def counters() -> dict[str, float]:
        stats = getattr(memory, "stats", {})
        values = {key: float(value) for key, value in (stats.items() if isinstance(stats, dict) else [])}
        values.update(actor_calls=float(actor_chat.calls), actor_prompt_tokens=float(actor_chat.prompt_tokens),
                      actor_completion_tokens=float(actor_chat.completion_tokens))
        return values

    recorder.counters_provider = counters
    reasoning_type, _ = module_map("io", None)
    reasoning = reasoning_type(llm_model=actor_chat)
    manager.mas.add_observer(recorder)
    manager.mas.build_system(reasoning, memory, manager.env, manager.mas_config)
    recorder.log(
        f"Episode checkpoint resume: completed={len(recorder.cases)}, "
        f"remaining={len(manager.tasks)}, path={checkpoint_path}"
    )
    try:
        if args.task == "officebench":
            _run_officebench_tasks(manager)
        else:
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
        "actor_calls": float(actor_chat.calls),
        **memory_stats,
        "sop_json_valid_rate": float(memory_stats.get("sop_json_valid", 0.0)) / calls if calls else 0.0,
        "sop_candidate_pass_rate": (
            float(memory_stats.get("sop_candidates_passed", 0.0)) / candidates if candidates else 0.0
        ),
    }
    metadata = {
        "adapter": "adapters/sepm_gmemory_adapter.py",
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
        "sepm_db": str(database_path),
        "actor_model": args.actor_model,
        "sop_model": args.sop_model,
        "actor_stop_policy": "relay-incompatible newline stop omitted; env selects first line",
        "actor_max_tokens": actor_chat.default_max_tokens,
        "episode_checkpoint": str(checkpoint_path),
        "resumed_case_count": len(checkpoint["cases"]),
        "scratch_policy": "adapter-owned work_dir/tmp; old env closed before reset",
        "causal_policy": (
            "candidates promote only via recorded reproduction trials against the unassisted "
            "same-task-type baseline; attribution requires executed candidate steps"
        ),
        "alignment_policy": "per-step blackboard alignment; corrections appended to the next actor call",
        "task_order": "adapter order" if args.seed == 0 else f"random.Random({args.seed}).shuffle",
        "frozen_pool": os.environ.get("SEPM_FREEZE_POOL") == "1",
        "task_type_filter": {"include": os.environ.get("SEPM_TASK_TYPES", ""),
                             "exclude": os.environ.get("SEPM_EXCLUDE_TASK_TYPES", "")},
        "pool_statistics": _pool_statistics(memory) if memory_method == "sepm" else None,
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
    from sepm.evaluation_adapter import EvaluationContext

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
