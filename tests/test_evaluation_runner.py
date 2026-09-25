"""统一评测矩阵、断点状态和单 case 运行的离线测试。"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from team_memory.evaluation_adapter import EvaluationContext
from team_memory.evaluation_runner import (
    EvalCell,
    EvaluationState,
    _state_db_path,
    build_report,
    expand_matrix,
    import_legacy_results,
    load_env_file,
    load_matrix,
    main,
    run_matrix,
    source_reproduction_commands,
    smoke_llm,
)
from team_memory.external_results import extract_metrics, load_native_result


class EvaluationRunnerTests(unittest.TestCase):
    def test_evaluation_state_uses_wal_for_parallel_case_workers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = EvaluationState(Path(directory) / "state.db")
            try:
                journal_mode = state.connection.execute("PRAGMA journal_mode").fetchone()[0]
                busy_timeout = state.connection.execute("PRAGMA busy_timeout").fetchone()[0]
            finally:
                state.close()
        self.assertEqual(journal_mode, "wal")
        self.assertEqual(busy_timeout, 30_000)

    def test_default_matrix_has_no_implicit_paper_experiment(self) -> None:
        config = load_matrix("evaluation_matrix.json")
        cells = expand_matrix(config)
        self.assertEqual(cells, [])

    def test_generality_uses_only_officially_wired_host_mas(self) -> None:
        config = load_matrix("evaluation_matrix.json")
        cells = expand_matrix(
            config,
            benchmark_filter="cross-benchmark-generality",
            task_filter="alfworld",
            method_filter="gmemory",
            actor_model_filter="gpt-5-mini",
        )
        self.assertEqual({cell.mas_framework for cell in cells}, {"autogen", "dylan"})
        self.assertTrue(all(cell.sop_model == "not-applicable" for cell in cells))

    def test_sop_model_sweep_keeps_actor_and_mas_fixed(self) -> None:
        config = load_matrix("evaluation_matrix.json")
        cells = expand_matrix(
            config,
            benchmark_filter="sop-model-sensitivity",
            task_filter="alfworld",
            method_filter="team-memory",
            actor_model_filter="gpt-5-mini",
        )
        self.assertEqual(len(cells), 144)
        self.assertEqual({cell.actor_model for cell in cells}, {"gpt-5-mini"})
        self.assertEqual({cell.mas_framework for cell in cells}, {"autogen"})

    def test_sop_model_filter_keeps_officially_wired_e1_methods(self) -> None:
        config = load_matrix("evaluation_matrix.json")
        cells = expand_matrix(
            config,
            benchmark_filter="cross-benchmark-generality",
            task_filter="alfworld",
            actor_model_filter="gpt-5-mini",
            sop_model_filter="gpt-5-mini",
            mas_filter="autogen",
        )
        self.assertEqual({cell.memory_method for cell in cells}, {
            "no-memory",
            "gmemory",
            "team-memory",
        })

    def test_case_id_filter_selects_single_paired_case(self) -> None:
        config = load_matrix("evaluation_matrix.json")
        cells = expand_matrix(
            config,
            benchmark_filter="cross-benchmark-generality",
            task_filter="alfworld",
            method_filter="team-memory",
            actor_model_filter="gpt-5-mini",
            sop_model_filter="gpt-5-mini",
            mas_filter="autogen",
            case_id="3",
        )
        self.assertEqual(len(cells), 1)
        self.assertTrue(all(cell.case_id == "3" for cell in cells))

    def test_seed_filter_keeps_fixed_seed_zero_resume_unit(self) -> None:
        config = load_matrix("evaluation_matrix.json")
        cells = expand_matrix(
            config,
            benchmark_filter="cross-benchmark-generality",
            task_filter="alfworld",
            method_filter="team-memory",
            actor_model_filter="gpt-5-mini",
            sop_model_filter="gpt-5-mini",
            mas_filter="autogen",
            seed_filter=0,
            case_id="3",
        )
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0].seed, 0)
        self.assertEqual(cells[0].case_id, "3")

    def test_cli_returns_nonzero_for_not_configured_cells(self) -> None:
        config = load_matrix("evaluation_matrix.json")
        with (
            patch("team_memory.evaluation_runner.load_env_file"),
            patch("team_memory.evaluation_runner.load_matrix", return_value=config),
            patch("team_memory.evaluation_runner.expand_matrix", return_value=[]),
            patch(
                "team_memory.evaluation_runner.run_matrix",
                return_value={"not_configured": 1},
            ),
        ):
            self.assertEqual(main(["run"]), 1)

    def test_env_file_does_not_override_server_secret_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("OPENAI_API_KEY=file-secret\nOPENAI_BASE_URL=https://relay/v1\n")
            with patch.dict(os.environ, {"OPENAI_API_KEY": "server-secret"}, clear=True):
                load_env_file(path)
                self.assertEqual(os.environ["OPENAI_API_KEY"], "server-secret")
                self.assertEqual(os.environ["OPENAI_BASE_URL"], "https://relay/v1")

    def test_single_llm_smoke_uses_relay_without_returning_key(self) -> None:
        config = load_matrix("evaluation_matrix.json")

        class FakeCompletions:
            def create(self, **kwargs):
                self.request = kwargs
                usage = types.SimpleNamespace(
                    prompt_tokens=2, completion_tokens=1, total_tokens=3
                )
                message = types.SimpleNamespace(content="OK")
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=message)], usage=usage
                )

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.options = kwargs
                self.chat = types.SimpleNamespace(completions=FakeCompletions())

        fake_module = types.SimpleNamespace(OpenAI=FakeOpenAI)
        environment = {
            "OPENAI_API_KEY": "relay-secret",
            "OPENAI_BASE_URL": "https://relay.example/v1",
        }
        with (
            patch.dict(sys.modules, {"openai": fake_module}),
            patch.dict(os.environ, environment, clear=True),
        ):
            result = smoke_llm(config, "gpt-5-mini", "只回复 OK")
        self.assertEqual(result["text"], "OK")
        self.assertEqual(result["usage"]["total_tokens"], 3)
        self.assertNotIn("relay-secret", json.dumps(result))

    def test_matrix_rejects_literal_provider_secret(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original = json.loads(Path("evaluation_matrix.json").read_text(encoding="utf-8"))
            original["provider"]["api_key"] = "must-not-be-committed"
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(original), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "literal secrets"):
                load_matrix(path)

    def test_state_tracks_attempts_without_storing_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = EvaluationState(Path(directory) / "state.db")
            cell = EvalCell(
                benchmark="bench",
                task="task",
                memory_method="method",
                actor_model="actor-model",
                sop_model="sop-model",
                mas_framework="mas",
                seed=0,
                case_id="case",
            )
            state.start(cell, Path("result.json"), Path("run.log"))
            state.finish(cell.key, "success")
            self.assertEqual(state.status(cell.key), "success")
            dump = " ".join(
                str(value)
                for row in state.connection.execute("SELECT * FROM evaluation_runs")
                for value in row
            )
            self.assertNotIn("OPENAI_API_KEY", dump)
            state.close()

    def test_external_adapter_contract_writes_standard_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            environment = {
                "TEAM_MEMORY_EVAL_BENCHMARK": "cross-benchmark-generality",
                "TEAM_MEMORY_EVAL_TASK": "alfworld",
                "TEAM_MEMORY_EVAL_METHOD": "team-memory",
                "TEAM_MEMORY_EVAL_ACTOR_MODEL": "actor-model",
                "TEAM_MEMORY_EVAL_SOP_MODEL": "sop-model",
                "TEAM_MEMORY_EVAL_MAS": "autogen",
                "TEAM_MEMORY_EVAL_ABLATION": "full",
                "TEAM_MEMORY_EVAL_SEED": "7",
                "TEAM_MEMORY_EVAL_CASE_ID": "q-1",
                "TEAM_MEMORY_EVAL_OUTPUT": str(output),
            }
            with patch.dict(os.environ, environment, clear=True):
                context = EvaluationContext.from_environment()
                context.write_result({"accuracy": 1.0}, cases=[{"id": "q-1"}])
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["metrics"], {"accuracy": 1.0})
            self.assertEqual(payload["context"]["case_id"], "q-1")
            self.assertEqual(payload["context"]["benchmark"], "cross-benchmark-generality")

    def test_source_commands_include_current_paper_benchmark(self) -> None:
        config = load_matrix("evaluation_matrix.json")
        commands = source_reproduction_commands(config, "cross-benchmark-generality")
        self.assertEqual(len(commands), 1)
        repository = Path(commands[0]["repository"])
        self.assertEqual(repository, Path(".").resolve())
        self.assertIn("cross-benchmark-generality", commands[0]["commands"][0])

    def test_resume_skips_successful_case_cell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_matrix("evaluation_matrix.json")
            config["output_dir"] = str(Path(directory) / "output")
            config["state_db"] = str(Path(directory) / "state.db")
            cell = EvalCell(
                benchmark="cross-benchmark-generality",
                task="alfworld",
                memory_method="team-memory",
                actor_model="gpt-5-mini",
                sop_model="gpt-5-mini",
                mas_framework="autogen",
                seed=0,
                case_id="0",
            )
            with patch("team_memory.evaluation_runner.run_cell", return_value="success") as run_cell:
                first = run_matrix(config, [cell])
                second = run_matrix(config, [cell], resume=True)
            self.assertEqual(first, {"success": 1})
            self.assertEqual(second, {"success": 1})
            run_cell.assert_called_once()
            report = build_report(config)
            self.assertEqual(report["status_counts"], {"success": 1})

    def test_state_db_override_uses_isolated_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_matrix("evaluation_matrix.json")
            default_state = Path(directory) / "default.db"
            override_state = Path(directory) / "override.db"
            config["state_db"] = str(default_state)
            with patch.dict(os.environ, {"TEAM_MEMORY_EVAL_STATE_DB": str(override_state)}):
                state = EvaluationState(_state_db_path(config))
                try:
                    self.assertEqual(state.path, override_state)
                finally:
                    state.close()
            self.assertFalse(default_state.exists())
            self.assertTrue(override_state.exists())

    def test_import_legacy_full_episode_as_case_level_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "legacy"
            legacy.mkdir()
            payload = {
                "context": {
                    "benchmark": "cross-benchmark-generality",
                    "task": "alfworld",
                    "memory_method": "team-memory",
                    "actor_model": "gpt-5-mini",
                    "sop_model": "gpt-5-mini",
                    "mas_framework": "autogen",
                    "ablation": "full",
                    "seed": 0,
                },
                "cases": [
                    {"case_id": "0", "official_task_index": 0, "reward": 1.0, "done": True}
                ],
            }
            (legacy / "old.json").write_text(json.dumps(payload), encoding="utf-8")
            config = load_matrix("evaluation_matrix.json")
            config["output_dir"] = str(root / "output")
            config["state_db"] = str(root / "state.db")
            cells = expand_matrix(
                config,
                benchmark_filter="cross-benchmark-generality",
                task_filter="alfworld",
                method_filter="team-memory",
                actor_model_filter="gpt-5-mini",
                sop_model_filter="gpt-5-mini",
                mas_filter="autogen",
                ablation_filter="full",
                case_id="0",
            )
            summary = import_legacy_results(config, cells, legacy_results_dir=legacy)
            self.assertEqual(summary["imported"], 1)
            state = EvaluationState(config["state_db"])
            try:
                self.assertEqual(state.status(cells[0].key), "success")
            finally:
                state.close()

    def test_import_legacy_full_run_accepts_api_model_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "legacy"
            legacy.mkdir()
            current_config = load_matrix("evaluation_matrix.json")
            case_id = current_config["benchmarks"][1]["case_ids"]["alfworld"][0]
            payload = {
                "context": {
                    "benchmark": "sop-model-sensitivity",
                    "task": "alfworld",
                    "memory_method": "team-memory",
                    "actor_model": "gpt-5-mini",
                    "sop_model": "Qwen/Qwen3.5-27B",
                    "mas_framework": "autogen",
                    "ablation": "full",
                    "seed": 0,
                    "case_id": case_id,
                },
                "metrics": {
                    "primary_score": 0.75,
                    "case_count": 134,
                    "sop_safety_rejections": 2,
                },
                "cases": [{"case_id": case_id, "done": True}],
            }
            (legacy / "old-full.json").write_text(json.dumps(payload), encoding="utf-8")
            config = current_config
            config["output_dir"] = str(root / "output")
            config["state_db"] = str(root / "state.db")
            cells = expand_matrix(
                config,
                benchmark_filter="sop-model-sensitivity",
                task_filter="alfworld",
                method_filter="team-memory",
                actor_model_filter="gpt-5-mini",
                sop_model_filter="qwen3.5-27b",
                mas_filter="autogen",
                ablation_filter="full",
                case_id=case_id,
            )
            self.assertEqual(len(cells), 1)
            summary = import_legacy_results(config, cells, legacy_results_dir=legacy)
            self.assertEqual(summary["imported"], 1)
            state = EvaluationState(config["state_db"])
            try:
                imported = [cell for cell in cells if state.status(cell.key) == "success"]
                self.assertEqual(len(imported), 1)
            finally:
                state.close()

    def test_external_job_uses_case_condition_and_repository_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "external" / "bench"
            repository.mkdir(parents=True)
            matrix = json.loads(Path("evaluation_matrix.json").read_text(encoding="utf-8"))
            matrix["output_dir"] = "output"
            matrix["state_db"] = "output/state.db"
            matrix["benchmarks"] = [
                {
                    "id": "external-test",
                    "enabled": True,
                    "uses_llm": True,
                    "runner": "external",
                    "repository": "external/bench",
                    "tasks": ["all"],
                    "case_ids": ["case-1"],
                    "methods": ["team-memory"],
                    "mas_frameworks": ["autogen"],
                    "actor_models": ["gpt-5-mini"],
                    "sop_models": ["gpt-5-mini"],
                    "ablations": ["sop-only"],
                    "seeds": [0],
                    "method_jobs": {
                        "team-memory": {
                            "conda_env": "bench-env",
                            "command": ["python", "adapter.py", "--out", "{output}"],
                        }
                    },
                }
            ]
            config_path = root / "matrix.json"
            config_path.write_text(json.dumps(matrix), encoding="utf-8")
            config = load_matrix(config_path)
            cell = expand_matrix(config)[0]

            def complete(command, **kwargs):
                output = Path(kwargs["env"]["TEAM_MEMORY_EVAL_OUTPUT"])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text('{"metrics":{"accuracy":1.0}}', encoding="utf-8")
                return types.SimpleNamespace(returncode=0)

            with patch("team_memory.evaluation_runner.subprocess.run", side_effect=complete) as run:
                with patch.dict(os.environ, {"OPENAI_API_KEY": "test-relay-key"}, clear=False):
                    summary = run_matrix(config, [cell])
            self.assertEqual(summary, {"success": 1})
            self.assertEqual(run.call_args.kwargs["cwd"], repository)
            self.assertEqual(run.call_args.kwargs["env"]["TEAM_MEMORY_EVAL_ABLATION"], "sop-only")
            self.assertEqual(run.call_args.kwargs["env"]["TEAM_MEMORY_EVAL_CASE_ID"], "case-1")

    def test_external_jsonl_metric_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native.jsonl"
            path.write_text(
                '{"question_id":"q1","autoeval_label":1}\n'
                '{"question_id":"q2","autoeval_label":0}\n',
                encoding="utf-8",
            )
            payload, cases = load_native_result(path, "jsonl")
            metrics = extract_metrics(payload, ["accuracy=autoeval_label:mean"])
            self.assertEqual(metrics, {"accuracy": 0.5})
            self.assertEqual(len(cases), 2)


if __name__ == "__main__":
    unittest.main()
