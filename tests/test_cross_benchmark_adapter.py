"""Cross-benchmark adapter safety checks."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adapters.team_memory_cross_benchmark_adapter import (
    _assert_gmemory_bridge_invoked,
    _runtime_identity_metadata,
    _write_manifest_smoke,
)
from team_memory.evaluation_adapter import EvaluationContext


class CrossBenchmarkAdapterTests(unittest.TestCase):
    def test_manifest_smoke_requires_explicit_environment_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context = EvaluationContext(
                benchmark="cross-benchmark-generality",
                task="webarena",
                memory_method="team-memory",
                actor_model="actor",
                sop_model="sop",
                mas_framework="autogen",
                seed=0,
                ablation="full",
                case_id="219",
                output_path=Path(directory) / "result.json",
            )
            args = type(
                "Args",
                (),
                {"task": "webarena", "case_id": "219"},
            )()
            with patch.dict("os.environ", {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "not paper evidence"):
                    _write_manifest_smoke(
                        context,
                        args=args,
                        project_root=Path(directory),
                        case={"case_id": "219"},
                    )

    def test_non_alfworld_dylan_rejects_metadata_only_host(self) -> None:
        args = type(
            "Args",
            (),
            {"task": "webarena", "mas": "dylan", "memory_method": "team-memory", "case_id": "219"},
        )()
        with self.assertRaisesRegex(RuntimeError, "no verified real"):
            _runtime_identity_metadata(args=args, task="webarena", official_entrypoint="external/webarena/run.py")

    def test_non_alfworld_gmemory_rejects_local_search_baseline(self) -> None:
        args = type(
            "Args",
            (),
            {"task": "officebench", "mas": "autogen", "memory_method": "gmemory", "case_id": "2-39/1"},
        )()
        with self.assertRaisesRegex(RuntimeError, "no verified real"):
            _runtime_identity_metadata(args=args, task="officebench", official_entrypoint="external/OfficeBench/agent_interact.py")

    def test_gmemory_result_requires_bridge_invocation(self) -> None:
        args = type("Args", (), {"memory_method": "gmemory"})()
        with self.assertRaisesRegex(RuntimeError, "bridge was never invoked"):
            _assert_gmemory_bridge_invoked(args, {}, Path("runtime.log"))


if __name__ == "__main__":
    unittest.main()
