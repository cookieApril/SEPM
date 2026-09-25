import json
import tempfile
import unittest
from pathlib import Path

from adapters.team_memory_gmemory_adapter import (
    EPISODE_CHECKPOINT_VERSION,
    OFFICIAL_INDEX_KEY,
    StructuredRecorder,
    _atomic_write_json,
    _install_resource_safe_reset,
    _load_episode_checkpoint,
    _normalize_actor_action,
    _official_case_id,
    _parse_json_object,
    _remaining_tasks,
    _select_cases,
    _string_list,
    _validate_official_data,
)


class GMemoryAdapterTests(unittest.TestCase):
    def test_string_list_preserves_scalar_as_one_item(self) -> None:
        self.assertEqual(_string_list("use valid actions", "applicability"), ["use valid actions"])
        self.assertEqual(_string_list(["one", 2], "exclusions"), ["one", "2"])
        with self.assertRaises(TypeError):
            _string_list({"bad": "shape"}, "applicability")

    def test_episode_checkpoint_round_trip_and_identity_gate(self) -> None:
        identity = {"task": "alfworld", "seed": 0}
        payload = {
            "version": EPISODE_CHECKPOINT_VERSION,
            "identity": identity,
            "cases": [{"official_task_index": 3, "done": True}],
            "actor_prompt_tokens": 10,
            "actor_completion_tokens": 2,
            "memory_stats": {},
            "complete": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            _atomic_write_json(path, payload)
            self.assertEqual(_load_episode_checkpoint(path, identity), payload)
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                _load_episode_checkpoint(path, {"task": "alfworld", "seed": 1})

    def test_remaining_tasks_skips_only_completed_official_indices(self) -> None:
        tasks = [
            {OFFICIAL_INDEX_KEY: 4, "task": "a"},
            {OFFICIAL_INDEX_KEY: 7, "task": "b"},
        ]
        remaining = _remaining_tasks(tasks, [{"official_task_index": 4}])
        self.assertEqual(remaining, [tasks[1]])
        with self.assertRaisesRegex(RuntimeError, "outside the selected dataset"):
            _remaining_tasks(tasks, [{"official_task_index": 99}])

    def test_recorder_checkpoint_resumes_after_completed_episode(self) -> None:
        class Delegate:
            def task_begin(self, task_id, task_config) -> None:
                pass

            def task_end(self, reward, done) -> None:
                pass

        identity = {"task": "alfworld", "seed": 0}
        tasks = [
            {OFFICIAL_INDEX_KEY: 4, "task": "a"},
            {OFFICIAL_INDEX_KEY: 7, "task": "b"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"

            def checkpoint(cases) -> None:
                _atomic_write_json(
                    path,
                    {
                        "version": EPISODE_CHECKPOINT_VERSION,
                        "identity": identity,
                        "cases": cases,
                        "actor_prompt_tokens": 0,
                        "actor_completion_tokens": 0,
                        "memory_stats": {},
                        "complete": False,
                    },
                )

            recorder = StructuredRecorder(Delegate(), checkpoint_callback=checkpoint)
            recorder.task_begin(0, tasks[0])
            recorder.task_end(1.0, True)

            restored = _load_episode_checkpoint(path, identity)
            self.assertEqual(restored["cases"][0]["official_task_index"], 4)
            self.assertEqual(_remaining_tasks(tasks, restored["cases"]), [tasks[1]])

    def test_resource_safe_reset_closes_previous_environment(self) -> None:
        events: list[str] = []

        class Current:
            def close(self) -> None:
                events.append("close")

        class Environment:
            env = Current()

            def reset(self) -> None:
                events.append("reset")

        environment = Environment()
        _install_resource_safe_reset(environment)
        environment.reset()
        self.assertEqual(events, ["close", "reset"])

    def test_official_case_id_uses_nested_alfworld_gamefile(self) -> None:
        task = {"env_kwargs": {"gamefile": "data/alfworld/example/game.tw-pddl"}}
        self.assertEqual(_official_case_id(task, 7), "data/alfworld/example/game.tw-pddl")

    def test_normalize_actor_action_removes_relay_formatting(self) -> None:
        self.assertEqual(_normalize_actor_action("go to desk 1.\nExplanation"), "go to desk 1")
        self.assertEqual(_normalize_actor_action("8. use desklamp 1."), "use desklamp 1")
        self.assertEqual(_normalize_actor_action("Act 3: > take bowl 1 from desk 2."), "take bowl 1 from desk 2")
        self.assertEqual(_normalize_actor_action("> think: inspect the desk."), "think: inspect the desk.")

    def test_normalize_actor_action_rejects_empty_text(self) -> None:
        with self.assertRaises(RuntimeError):
            _normalize_actor_action("\n  \n")

    def test_parse_json_object_accepts_plain_and_fenced_json(self) -> None:
        expected = {"atomic_steps": ["inspect", "act"]}
        self.assertEqual(_parse_json_object(json.dumps(expected)), expected)
        self.assertEqual(_parse_json_object(f"```json\n{json.dumps(expected)}\n```"), expected)

    def test_parse_json_object_rejects_non_object(self) -> None:
        with self.assertRaises(TypeError):
            _parse_json_object("[]")

    def test_select_cases_uses_official_index_or_id(self) -> None:
        rows = [{"id": "alpha"}, {"id": "beta"}]
        self.assertEqual(_select_cases(rows, "1"), [rows[1]])
        self.assertEqual(_select_cases(rows, "alpha"), [rows[0]])
        with self.assertRaises(ValueError):
            _select_cases(rows, "missing")

    def test_alfworld_preflight_lists_missing_official_data(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(FileNotFoundError, "json_2.1.1/valid_unseen"),
        ):
            _validate_official_data(Path(directory), "alfworld")


if __name__ == "__main__":
    unittest.main()
