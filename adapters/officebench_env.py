"""OfficeBench behind G-Memory's ``BaseEnv`` interface (AutoGen/DyLAN hosts).

G-Memory's AutoGen and DyLAN workflows drive any environment exposing
``set_env``/``reset``/``step``/``process_action``/``feedback``/``max_trials``. This
adapter runs the official OfficeBench Docker environment (``utils.env.OfficeAgentEnv``),
shows agents OfficeBench's own system message and per-app action menu
(``utils.policies``), and scores each task with OfficeBench's evaluator
(``evaluation.evaluate_output``). Requires Docker and the ``officebench`` image.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


def officebench_tasks(office_root: Path) -> list[dict[str, Any]]:
    """All 300 subtasks in OfficeBench's evaluation order (``evaluation.py``)."""
    rows = []
    for config_path in (office_root / "tasks").glob("*/subtasks/*.json"):
        task_dir, subtask_id = config_path.parts[-3], config_path.stem
        rows.append((tuple(map(int, task_dir.split("-"))) + (int(subtask_id),), task_dir, subtask_id, config_path))
    rows.sort()
    tasks = []
    for _, task_dir, subtask_id, config_path in rows:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        tasks.append({
            "id": f"{task_dir}/{subtask_id}",
            "task": config["task"],
            "task_dir": task_dir,
            "subtask_id": subtask_id,
            "config_path": str(config_path),
            "task_type": f"{task_dir.split('-')[0]}-app",
            "env_name": "officebench",
        })
    return tasks


@contextlib.contextmanager
def _cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class OfficeBenchEnv:
    """One Docker container per task, reset from the task's testbed."""

    def __init__(self, office_root: str | Path, *, docker_name: str = "officebench",
                 container_name: str = "officebench-sepm", max_trials: int = 20,
                 output_root: str | Path | None = None) -> None:
        self.root = Path(office_root).resolve()
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        with _cwd(self.root):
            import apps  # noqa: F401 - OfficeBench app registry
            from evaluation import evaluate_output
            from utils.env import OfficeAgentEnv
            from utils.policies import LLMPolicy, PROMPT_DICT
        self._env_cls, self._policy_cls, self._prompts = OfficeAgentEnv, LLMPolicy, PROMPT_DICT
        self._evaluate = evaluate_output
        self.docker_name, self.container_name = docker_name, container_name
        self.max_trials = max_trials
        self.output_root = Path(output_root or self.root / "sepm_outputs").resolve()
        self.env: Any = None
        self.task: dict[str, Any] = {}
        self.done = False
        self.finished = False

    # ---- G-Memory BaseEnv interface ----
    def set_env(self, task_config: dict[str, Any]) -> tuple[str, str]:
        self.close()
        self.task = task_config
        config = json.loads(Path(task_config["config_path"]).read_text(encoding="utf-8"))
        self.config = config
        with _cwd(self.root):
            self.env = self._env_cls(image_name=self.docker_name, container_name=self.container_name,
                                     task=config["task"], verbose=False)
            self.env.reset()
            self.env.prepare_docker_env(testbed_dir=str(self.root / "tasks" / task_config["task_dir"] / "testbed") + "/",
                                        app_dir=str(self.root / "apps") + "/")
        self.system_message = self._policy_cls.construct_system_message(None, config, self.env.available_apps)
        self.done = self.finished = False
        return config["task"], f"{self.system_message}\n\n{self._menu()}"

    def reset(self) -> None:
        self.done = False

    @staticmethod
    def process_action(action: str) -> str:
        if "{" in action and "}" in action:
            return action[action.find("{"): action.rfind("}") + 1]
        return action.strip()

    def step(self, action: str) -> tuple[str, float, bool]:
        action = self.process_action(action)
        with _cwd(self.root):
            self.env.exec_action(action)
        observation = str(self.env.observation)
        self.done = observation in {"Task finished", "Task failed"}
        self.finished = observation == "Task finished"
        reward = -1.0 if observation.startswith("Malformed action") else 0.0
        if not self.done:
            observation = f"{observation}\n{self._menu()}"
        return observation, reward, self.done

    def feedback(self) -> tuple[float, bool, str]:
        out = self.output_root / self.task["task_dir"] / self.task["subtask_id"]
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        with _cwd(self.root):
            self.env.cache_docker_status(local_cache_dir=str(out))
            passed = bool(self._evaluate(self.task["task_dir"], self.task["subtask_id"], str(out / "testbed")))
        message = "OfficeBench evaluation passed." if passed else "OfficeBench evaluation failed."
        return (1.0 if passed else 0.0), passed, message

    def close(self) -> None:
        if self.env is not None:
            with contextlib.suppress(Exception):
                self.env.close()
        self.env = None

    # ---- helpers ----
    def _menu(self) -> str:
        """OfficeBench's own instruction block for the current app (``utils.policies``)."""
        env = self.env
        apps_list = list(env.available_apps.keys())
        if env.current_app is None:
            return ("##Available apps: " + str(apps_list) + "\n##Instruction:\n"
                    " - choose an app from the avaiblable apps: {'app': 'system', 'action': 'switch_app', "
                    "'target_app': [THE_APP_YOU_CHOOSE]}\n##Command:")
        import apps as office_apps

        detailed = "".join(
            f" - {office_apps.AVAILABLE_ACTIONS[env.current_app][name].DEMO}\n" for name in env.get_available_actions()
        )
        others = [app for app in apps_list if app != env.current_app]
        return (f"##Current apps: {env.current_app}\n##Instruction: Choose one action from the list as the next step.\n"
                f"{detailed}"
                f" - switch to another app among {others}: {{'app': 'system', 'action': 'switch_app', 'target_app': [THE_APP_YOU_CHOOSE]}}\n"
                " - finish the task with your answer as None if the task is not a question: {'app': 'system', 'action': 'finish_task', 'answer': 'None'}\n"
                " - finish the task with your answer if the task is a question: {'app': 'system', 'action': 'finish_task', 'answer': [ANSWER]}\n"
                "##Command:")


OFFICEBENCH_INSTRUCTION = (
    "You operate office applications through JSON commands. Output exactly one command per step, "
    "as a single-line dict in the format shown under ##Instruction (no explanation)."
)
