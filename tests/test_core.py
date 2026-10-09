"""Regression tests for the core service.

Each test uses an isolated temporary SQLite database and covers divergence,
evidence conflicts, safety prechecks, first-success promotion, lease exclusion,
retrieval weights, optimistic concurrency, and experiment policy. The suite
prefers readable behavior tests over tests of private implementation details.
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from sepm.ablation.validator_policy import PromptGatePolicy, ValidationScenario, benchmark_policy
from sepm.config import MemoryConfig
from sepm.divergence import normalized_plan_distance
from sepm.evidence import EvidenceResolver
from sepm.models import (
    AgentProfile,
    BlackboardEntry,
    BlackboardKind,
    Evidence,
    EvidenceTier,
    PlanEdge,
    PlanNode,
    ProcedureGraph,
    ProcedureStep,
    ProposalOperation,
    ReproductionTrial,
    SOPCandidate,
    SOPMetadata,
    TaskPlan,
    Workspace,
)
from sepm.safety import SafetyValidator
from sepm.service import SEPMService
from sepm.storage import ConcurrencyError


def procedure(*steps: ProcedureStep) -> ProcedureGraph:
    """Connect steps in input order to build a linear ProcedureGraph."""
    edges = [PlanEdge(source=steps[index].step_id, target=steps[index + 1].step_id) for index in range(len(steps) - 1)]
    return ProcedureGraph(steps=list(steps), edges=edges)


class SEPMTests(unittest.TestCase):
    """Validate cross-module business invariants through the real storage layer."""
    def setUp(self) -> None:
        """Create an isolated database and two base agents for each case."""
        self.tempdir = tempfile.TemporaryDirectory()
        config = MemoryConfig(database_path=Path(self.tempdir.name) / "memory.db")
        self.service = SEPMService(config=config)
        self.service.register_agent(AgentProfile(agent_id="agent-a", role="task executor"))
        self.service.register_agent(AgentProfile(agent_id="agent-b", role="task executor"))

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def workspace(self, workspace_id: str, family_action: str = "search") -> Workspace:
        """Create a workspace fixture with a two-node canonical plan."""
        workspace = Workspace(
            workspace_id=workspace_id,
            main_goal="Find a safe restaurant and ask before booking",
            plan=TaskPlan(
                nodes=[PlanNode(node_id="a", action=family_action), PlanNode(node_id="b", action="recommend")],
                edges=[PlanEdge(source="a", target="b")],
            ),
        )
        self.service.create_workspace(workspace)
        return workspace

    def test_plan_distance_detects_dependency_change(self) -> None:
        left = TaskPlan(
            nodes=[PlanNode(node_id="a", action="search"), PlanNode(node_id="b", action="rank")],
            edges=[PlanEdge(source="a", target="b")],
        )
        right = TaskPlan(
            nodes=[PlanNode(node_id="a", action="book"), PlanNode(node_id="b", action="pay")],
            edges=[PlanEdge(source="b", target="a")],
        )
        self.assertGreater(normalized_plan_distance(left, right), 0.5)

    def test_graph_models_reject_unknown_edge_endpoints(self) -> None:
        with self.assertRaises(ValueError):
            TaskPlan(
                nodes=[PlanNode(node_id="known", action="search")],
                edges=[PlanEdge(source="known", target="missing")],
            )

    def test_authoritative_state_beats_agent_inference(self) -> None:
        resolver = EvidenceResolver()
        timestamp = datetime.now(timezone.utc)
        result = resolver.resolve_world_state(
            "order.paid",
            [
                Evidence(tier=EvidenceTier.AGENT_INFERENCE, source="agent:A", value=False, confidence=0.9),
                Evidence(
                    tier=EvidenceTier.AUTHORITATIVE_STATE,
                    source="payment-api",
                    value=True,
                    observed_at=timestamp,
                    confidence=1.0,
                ),
            ],
        )
        self.assertTrue(result.resolved)
        self.assertIs(result.value, True)
        self.assertEqual(result.winning_evidence.source, "payment-api")

    def test_equal_tier_equal_time_conflict_requires_requery(self) -> None:
        timestamp = datetime.now(timezone.utc)
        result = EvidenceResolver().resolve_world_state(
            "order.paid",
            [
                Evidence(tier=EvidenceTier.AUTHORITATIVE_STATE, source="api-1", value=True, observed_at=timestamp),
                Evidence(tier=EvidenceTier.AUTHORITATIVE_STATE, source="api-2", value=False, observed_at=timestamp),
            ],
        )
        self.assertFalse(result.resolved)
        self.assertIn("re-query", result.next_action)

    def test_payment_shortcut_is_rejected_before_success_metrics(self) -> None:
        unsafe = SOPCandidate(
            operation=ProposalOperation.CREATE,
            procedure=procedure(ProcedureStep(instruction="Pay the order immediately", action_type="payment")),
            metadata=SOPMetadata(title="Fast payment", task_family="commerce"),
            source_workspace_ids=["trace-success"],
            state_verified=True,
            causal_confidence=0.99,
            claimed_benefit=100.0,
        )
        decision = SafetyValidator().validate(unsafe)
        self.assertFalse(decision.allowed)
        self.assertTrue(any("confirmation" in violation for violation in decision.violations))

    def test_service_detects_world_state_conflict(self) -> None:
        self.workspace("w-state")
        for agent, value in (("agent-a", False), ("agent-b", True)):
            self.service.append_blackboard(
                BlackboardEntry(
                    workspace_id="w-state",
                    agent_id=agent,
                    kind=BlackboardKind.STATE_CLAIM,
                    task="Find a safe restaurant and ask before booking",
                    goal="Find a safe restaurant and ask before booking",
                    state_key="booking.confirmed",
                    state_value=value,
                )
            )
        report = self.service.detect_divergence("w-state", "agent-a")
        self.assertIn("booking.confirmed", report.conflicting_state_keys)
        self.assertTrue(report.requires_alignment)

    def test_first_successful_validation_immediately_promotes(self) -> None:
        self.workspace("w-one")
        candidate = SOPCandidate(
            operation=ProposalOperation.CREATE,
            procedure=procedure(ProcedureStep(instruction="Search and rank verified options")),
            metadata=SOPMetadata(title="Verified search", task_family="search"),
            source_workspace_ids=["w-one"],
            state_verified=True,
            causal_confidence=0.9,
        )
        stored, safety = self.service.propose_sop(candidate)
        self.assertTrue(safety.allowed)
        self.service.record_reproduction(
            ReproductionTrial(
                candidate_id=stored.candidate_id,
                workspace_id="w-one",
                task_family="search",
                environment_fingerprint="env-1",
                success=True,
                baseline_reward=0.2,
                candidate_reward=0.9,
                cost=0.1,
                state_verified=True,
                causal_supported=True,
            )
        )
        promoted = self.service.store.get_candidate(stored.candidate_id)
        self.assertEqual(promoted.status.value, "promoted")
        sop = self.service.store.get_sop(stored.candidate_id)
        self.assertEqual(sop.success_count, 1)
        decision = self.service.promote_candidate(stored.candidate_id)
        self.assertTrue(decision.promoted)
        self.assertEqual(decision.reasons, ["candidate already promoted"])
        validated, _ = self.service.validate_candidate(stored.candidate_id)
        self.assertEqual(validated.status.value, "promoted")

    def test_unsuccessful_validation_does_not_write_sop(self) -> None:
        self.workspace("w-failed")
        candidate = SOPCandidate(
            operation=ProposalOperation.CREATE,
            procedure=procedure(ProcedureStep(instruction="Verify a failed result")),
            metadata=SOPMetadata(title="Failed validation", task_family="quality"),
            source_workspace_ids=["w-failed"],
            state_verified=True,
            causal_confidence=0.9,
        )
        stored, _ = self.service.propose_sop(candidate)
        self.service.record_reproduction(
            ReproductionTrial(
                candidate_id=stored.candidate_id,
                workspace_id="w-failed",
                task_family="quality",
                environment_fingerprint="env-failed",
                success=False,
                baseline_reward=0.5,
                candidate_reward=0.2,
                cost=0.1,
                state_verified=True,
                causal_supported=True,
            )
        )
        _, total = self.service.store.list_active_sops()
        self.assertEqual(total, 0)

    def test_candidate_queue_lease_prevents_double_claim(self) -> None:
        candidate = SOPCandidate(
            operation=ProposalOperation.CREATE,
            procedure=procedure(ProcedureStep(instruction="Verify a result")),
            metadata=SOPMetadata(title="Lease test", task_family="test"),
            source_workspace_ids=["trace"],
        )
        stored, _ = self.service.propose_sop(candidate)
        claimed = self.service.store.claim_candidate("worker-a", lease_seconds=60)
        self.assertEqual(claimed.candidate_id, stored.candidate_id)
        self.assertIsNone(self.service.store.claim_candidate("worker-b", lease_seconds=60))
        self.assertFalse(self.service.store.release_candidate(stored.candidate_id, "worker-b"))
        self.assertTrue(self.service.store.release_candidate(stored.candidate_id, "worker-a"))
        self.assertEqual(self.service.store.claim_candidate("worker-b").candidate_id, stored.candidate_id)

    def test_first_validation_promotes_and_retrieves(self) -> None:
        self.workspace("w-a")
        self.workspace("w-b", family_action="lookup")
        candidate = SOPCandidate(
            operation=ProposalOperation.CREATE,
            procedure=procedure(
                ProcedureStep(instruction="Query the authoritative catalog"),
                ProcedureStep(instruction="Rank only verified candidates"),
            ),
            metadata=SOPMetadata(
                title="Evidence-first ranking",
                task_family="retrieval",
                applicability=["catalog search", "document lookup"],
                tags=["verification", "ranking"],
            ),
            source_workspace_ids=["w-a", "w-b"],
            state_verified=True,
            causal_confidence=0.9,
        )
        stored, _ = self.service.propose_sop(candidate)
        self.service.record_reproduction(
            ReproductionTrial(
                candidate_id=stored.candidate_id,
                workspace_id="w-a",
                task_family="catalog-search",
                environment_fingerprint="catalog-v1",
                success=True,
                baseline_reward=0.3,
                candidate_reward=0.9,
                cost=0.1,
                state_verified=True,
                causal_supported=True,
            )
        )
        decision = self.service.promote_candidate(stored.candidate_id)
        self.assertTrue(decision.promoted)
        self.assertEqual(decision.sop.version, 1)
        results = self.service.retrieve_sops("rank verified catalog results", limit=3)
        self.assertEqual(results[0].sop.sop_id, decision.sop.sop_id)
        self.assertAlmostEqual(sum(results[0].weights.values()), 1.0)
        planned = self.service.retrieve_sops(
            "verified results",
            query_plan=TaskPlan(
                nodes=[
                    PlanNode(node_id=verify_id, action="generic")
                    for verify_id in (decision.sop.procedure.steps[0].step_id, decision.sop.procedure.steps[1].step_id)
                ],
                edges=decision.sop.procedure.edges,
            ),
        )
        self.assertIn("bm25_relevance", planned[0].features)
        self.assertNotIn("graph_relevance", planned[0].features)
        before = self.service.router.weights("routine task")["bm25"]
        self.service.router.learn("routine task", "bm25", learning_rate=0.2)
        after = self.service.router.weights("routine task")["bm25"]
        self.assertGreater(after, before)

    def test_optimistic_concurrency_blocks_stale_update(self) -> None:
        self.workspace("w-a")
        self.workspace("w-b")
        # Reuse a promoted SOP from the helper path.
        candidate = SOPCandidate(
            operation=ProposalOperation.CREATE,
            procedure=procedure(ProcedureStep(instruction="Verify the artifact")),
            metadata=SOPMetadata(title="Verify", task_family="quality"),
            source_workspace_ids=["w-a", "w-b"],
            state_verified=True,
            causal_confidence=0.9,
        )
        stored, _ = self.service.propose_sop(candidate)
        self.service.record_reproduction(
            ReproductionTrial(
                candidate_id=stored.candidate_id,
                workspace_id="w-a",
                task_family="quality",
                environment_fingerprint="env-0",
                success=True,
                baseline_reward=0,
                candidate_reward=1,
                cost=0,
                state_verified=True,
                causal_supported=True,
            )
        )
        sop = self.service.promote_candidate(stored.candidate_id).sop
        stale_copy = sop.model_copy(update={"version": 2})
        with self.assertRaises(ConcurrencyError):
            self.service.store.commit_sop(stale_copy, expected_version=999)

    def test_prompt_policy_never_accepts_unsafe_scenario(self) -> None:
        scenarios = [
            ValidationScenario("bad", False, True, True, True, True),
            ValidationScenario("good", True, True, True, True, True),
        ]
        metrics = benchmark_policy(PromptGatePolicy(), scenarios)
        self.assertEqual(metrics["unsafe_accept_rate"], 0.0)
        self.assertEqual(metrics["promotion_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
