"""Tests for benchmark host and memory integration bridges."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from team_memory.integrations import GMemoryBridge, GMemorySnapshotError, HostRuntimeBridge
from team_memory.benchmark_runtime import TeamMemoryBenchmarkRuntime


class IntegrationBridgeTests(unittest.TestCase):
    def test_frozen_gmemory_snapshot_retrieves_and_rejects_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "nodes.json").write_text(
                json.dumps(
                    [
                        {
                            "node_id": "n1",
                            "task": "buy a soap bottle",
                            "trajectory": "search, inspect cart, verify",
                            "insight": "confirm state before checkout",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (snapshot / "manifest.json").write_text(
                json.dumps({"snapshot_id": "snap1", "frozen": True}),
                encoding="utf-8",
            )
            evidence = root / "evidence.json"
            bridge = GMemoryBridge.open_snapshot(snapshot, read_only=True)
            bridge.evidence_path = evidence
            retrievals = bridge.retrieve(
                task="buy soap bottle",
                observation="cart page",
                agent_role="browser agent",
                top_k=2,
            )
            self.assertEqual([item.node_id for item in retrievals], ["n1"])
            self.assertIn("Relevant G-Memory", bridge.format_for_actor(retrievals))
            self.assertTrue(evidence.is_file())
            with self.assertRaises(GMemorySnapshotError):
                bridge.record_execution_event(action="mutate")

    def test_dylan_host_identity_differs_from_autogen(self) -> None:
        # This exercises metadata formatting, not execution of the upstream host.
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            upstream = project_root / "external/GMemory/tasks/mas_workflow/dylan"
            upstream.mkdir(parents=True)
            (upstream / "dylan.py").write_text("# metadata fixture")
            (upstream / "neuron.py").write_text("# metadata fixture")
            autogen = HostRuntimeBridge.build(mas="autogen", task="webarena", project_root=project_root)
            dylan = HostRuntimeBridge.build(mas="dylan", task="webarena", project_root=project_root)
            self.assertNotEqual(autogen.topology_hash, dylan.topology_hash)

    def test_unknown_host_fails_before_runtime_call(self) -> None:
        with self.assertRaisesRegex(Exception, "unsupported host"):
            HostRuntimeBridge.build(
                mas="renamed-autogen",
                task="webarena",
                project_root=Path(__file__).resolve().parents[1],
            )

    def test_dylan_gmemory_runtime_composes_host_and_memory_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "nodes.json").write_text(
                json.dumps(
                    [
                        {
                            "node_id": "n1",
                            "task": "inspect map distance",
                            "trajectory": "open map and verify route distance",
                            "insight": "keep map evidence visible before final answer",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (snapshot / "manifest.json").write_text(
                json.dumps({"snapshot_id": "snap-compose", "frozen": True}),
                encoding="utf-8",
            )
            output = root / "result.json"
            with patch.dict(
                "os.environ",
                {
                    "TEAM_MEMORY_PROJECT_ROOT": str(Path(__file__).resolve().parents[1]),
                    "TEAM_MEMORY_GMEMORY_SNAPSHOT": str(snapshot),
                    "TEAM_MEMORY_EVAL_BENCHMARK": "cross-benchmark-generality",
                    "TEAM_MEMORY_EVAL_TASK": "webarena",
                    "TEAM_MEMORY_EVAL_METHOD": "gmemory",
                    "TEAM_MEMORY_EVAL_ACTOR_MODEL": "gpt-5-mini",
                    "TEAM_MEMORY_EVAL_SOP_MODEL": "not-applicable",
                    "TEAM_MEMORY_EVAL_MAS": "dylan",
                    "TEAM_MEMORY_EVAL_SEED": "0",
                    "TEAM_MEMORY_EVAL_ABLATION": "full",
                    "TEAM_MEMORY_EVAL_CASE_ID": "219",
                    "TEAM_MEMORY_EVAL_OUTPUT": str(output),
                },
                clear=True,
            ):
                with self.assertRaisesRegex(GMemorySnapshotError, "no upstream G-Memory"):
                    TeamMemoryBenchmarkRuntime.from_environment(main_goal="inspect map distance")


if __name__ == "__main__":
    unittest.main()
