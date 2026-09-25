"""Runtime hook tests for official benchmark integrations."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from team_memory.benchmark_runtime import TeamMemoryBenchmarkRuntime
from team_memory.models import (
    ProcedureGraph,
    ProcedureStep,
    ProposalOperation,
    ReproductionTrial,
    SOPCandidate,
    SOPMetadata,
)


class BenchmarkRuntimeTests(unittest.TestCase):
    def _env(self, directory: str) -> dict[str, str]:
        return {
            "TEAM_MEMORY_EVAL_BENCHMARK": "component-ablation",
            "TEAM_MEMORY_EVAL_TASK": "webarena",
            "TEAM_MEMORY_EVAL_METHOD": "team-memory",
            "TEAM_MEMORY_EVAL_ACTOR_MODEL": "actor",
            "TEAM_MEMORY_EVAL_SOP_MODEL": "sop",
            "TEAM_MEMORY_EVAL_MAS": "autogen",
            "TEAM_MEMORY_EVAL_ABLATION": "full",
            "TEAM_MEMORY_EVAL_SEED": "7",
            "TEAM_MEMORY_EVAL_CASE_ID": "219",
            "TEAM_MEMORY_EVAL_OUTPUT": str(Path(directory) / "result.json"),
            "TEAM_MEMORY_DB": str(Path(directory) / "memory.db"),
        }

    def test_runtime_records_loop_events_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, self._env(directory), clear=True):
                runtime = TeamMemoryBenchmarkRuntime.from_environment(
                    main_goal="Book a verified item only after authorization",
                    plan_actions=["search item", "verify authorization", "checkout"],
                    plan_edges=[(0, 1), (1, 2)],
                )
                runtime.start_agent(
                    "agent-a",
                    "web executor",
                    current_task="Book a verified item only after authorization",
                )
                runtime.observe("agent-a", "cart is empty", state_key="cart", state_value="empty")
                action = runtime.before_action(
                    "agent-a",
                    "checkout now",
                    goal="Book a different item",
                )
                runtime.after_action(
                    "agent-a",
                    "payment API rejected authorization",
                    state_key="payment",
                    state_value="rejected",
                    authoritative=True,
                )

                metrics = runtime.finish_metrics({"primary_score": 0.0, "case_count": 1.0})

        self.assertTrue(action["requires_alignment"])
        self.assertEqual(metrics["team_memory_enabled"], 1.0)
        self.assertEqual(metrics["case_count"], 1.0)
        self.assertGreaterEqual(metrics["blackboard_entries"], 3.0)
        self.assertGreaterEqual(metrics["divergence_events"], 1.0)
        self.assertEqual(metrics["recovery_verified_success_count"], 0.0)
        self.assertEqual(metrics["divergence_recovery"], 0.0)

    def test_recovery_requires_explicit_verified_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, self._env(directory), clear=True):
                runtime = TeamMemoryBenchmarkRuntime.from_environment(
                    main_goal="Keep the verified account state",
                    plan_actions=["observe", "repair", "verify"],
                    plan_edges=[(0, 1), (1, 2)],
                )
                runtime.start_agent(
                    "agent-a",
                    "web executor",
                    current_task="Keep the verified account state",
                )
                action = runtime.before_action(
                    "agent-a",
                    "overwrite account state",
                    goal="Use an unverified account state",
                )
                event_id = action["active_divergence_event_id"]
                runtime.record_recovery_proposed(
                    "agent-a",
                    "Reload authoritative account page",
                    target="account_state",
                    event_id=event_id,
                )
                runtime.record_recovery_executed(
                    "agent-a",
                    "navigate to account page and refresh state",
                    event_id=event_id,
                )
                runtime.record_recovery_verified(
                    "agent-a",
                    "official page shows verified account state",
                    success=True,
                    event_id=event_id,
                )
                metrics = runtime.finish_metrics()

        self.assertIsNotNone(event_id)
        self.assertEqual(metrics["conflict_detected_count"], 1.0)
        self.assertEqual(metrics["recovery_proposed_count"], 1.0)
        self.assertEqual(metrics["recovery_executed_count"], 1.0)
        self.assertEqual(metrics["recovery_verified_count"], 1.0)
        self.assertEqual(metrics["recovery_verified_success_count"], 1.0)
        self.assertEqual(metrics["divergence_recovery"], 1.0)

    def test_runtime_retrieves_existing_sop_for_prompt_injection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = self._env(directory)
            with patch.dict(os.environ, env, clear=True):
                runtime = TeamMemoryBenchmarkRuntime.from_environment(
                    main_goal="Search then verify before checkout"
                )
                runtime.start_agent(
                    "agent-a",
                    "web executor",
                    current_task="checkout with verification",
                )
                candidate, _ = runtime.service.propose_sop(
                    SOPCandidate(
                        operation=ProposalOperation.CREATE,
                        procedure=ProcedureGraph(
                            steps=[
                                ProcedureStep(instruction="Search for the target item"),
                                ProcedureStep(
                                    instruction="Verify authorization before checkout",
                                    requires_confirmation=True,
                                    safety_critical=True,
                                ),
                            ]
                        ),
                        metadata=SOPMetadata(
                            title="Verified checkout",
                            task_family="webarena",
                            safety_class="sensitive",
                        ),
                        source_workspace_ids=[runtime.workspace.workspace_id],
                        state_verified=True,
                        causal_confidence=1.0,
                        claimed_benefit=1.0,
                    )
                )
                runtime.service.record_reproduction(
                    ReproductionTrial(
                        candidate_id=candidate.candidate_id,
                        workspace_id=runtime.workspace.workspace_id,
                        task_family="webarena",
                        environment_fingerprint="test",
                        success=True,
                        baseline_reward=0.0,
                        candidate_reward=1.0,
                        cost=0.0,
                        state_verified=True,
                        causal_supported=True,
                    )
                )

                second = TeamMemoryBenchmarkRuntime.from_environment(
                    main_goal="Search then verify before checkout"
                )
                payload = second.start_agent(
                    "agent-b",
                    "web executor",
                    current_task="checkout with verification",
                )
                second.before_action("agent-b", "inspect item", used_sop_id=candidate.candidate_id)
                self.assertEqual(second.finish_metrics()["sop_reuse_success"], 0)
                second.record_sop_outcome(candidate.candidate_id, success=False)
                self.assertEqual(second.finish_metrics()["sop_reuse_success"], 0)
                self.assertEqual(second.finish_metrics()["sop_verified_outcome_count"], 1)

        self.assertIn("Verified checkout", payload["prompt_prefix"])
        self.assertEqual(second.finish_metrics()["sop_retrieval_count"], 1.0)


if __name__ == "__main__":
    unittest.main()
