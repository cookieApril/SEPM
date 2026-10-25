"""Behavioral regressions found while auditing the three research contributions."""
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from sepm.benchmark_runtime import SEPMBenchmarkRuntime
from sepm.config import MemoryConfig
from sepm.divergence import normalized_plan_distance
from sepm.evidence import EvidenceResolver
from sepm.models import (
    AgentProfile,
    BlackboardEntry,
    BlackboardKind,
    CandidateStatus,
    ConditionalWarning,
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
    SOPVersion,
    TaskPlan,
    Workspace,
)
from sepm.safety import SafetyValidator
from sepm.service import SEPMService
from sepm.storage import ConcurrencyError, NotFoundError


class ContributionRegressionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.service = SEPMService(Path(self.tmp.name) / "memory.db")
        self.service.create_workspace(Workspace(workspace_id="w", main_goal="Find a restaurant"))
        for agent in ("a", "b"):
            self.service.register_agent(AgentProfile(agent_id=agent, role="worker"))

    def entry(self, agent="a", **fields):
        self.service.append_blackboard(BlackboardEntry(
            workspace_id="w", agent_id=agent, kind=BlackboardKind.OBSERVATION, **fields))

    def candidate(self, **fields):
        values = {
            "operation": ProposalOperation.CREATE,
            "procedure": ProcedureGraph(steps=[ProcedureStep(instruction="Inspect result")]),
            "metadata": SOPMetadata(title="Inspect", task_family="inspect"),
            "source_workspace_ids": ["w"],
            "state_verified": True,
            "causal_confidence": 1,
            "claimed_benefit": 1,
        }
        values.update(fields)
        return SOPCandidate(**values)

    def trial(self, candidate, **fields):
        values = {
            "candidate_id": candidate.candidate_id,
            "workspace_id": "w",
            "task_family": "inspect",
            "environment_fingerprint": "env",
            "success": True,
            "baseline_reward": 0,
            "candidate_reward": 1,
            "cost": 0,
            "state_verified": True,
            "causal_supported": True,
        }
        values.update(fields)
        return ReproductionTrial(**values)

    def test_sparse_observation_preserves_goal_and_plan(self):
        plan = TaskPlan(nodes=[PlanNode(node_id="pay", action="pay")])
        self.entry(goal="Book and pay immediately", plan=plan)
        self.entry(observation="page loaded")
        report = self.service.detect_divergence("w", "a")
        self.assertTrue(report.goal_observed)
        self.assertTrue(report.plan_observed)
        self.assertIn("goal_divergence", report.reasons)

    def test_agent_start_does_not_reset_canonical_workspace(self):
        old = self.service.store.get_workspace("w")
        revised = old.model_copy(update={"main_goal": "Updated user goal", "goal_version": 2})
        self.service.store.put_workspace(revised)
        returned = self.service.create_workspace(old)
        self.assertEqual(returned.main_goal, "Updated user goal")

    def test_missing_goal_is_not_divergence(self):
        self.entry(task="Rank candidates", observation="done")
        report = self.service.detect_divergence("w", "a")
        self.assertFalse(report.goal_observed)
        self.assertFalse(report.plan_observed)
        self.assertFalse(report.requires_alignment)

    def test_state_update_is_not_permanent_conflict(self):
        self.entry(state_key="paid", state_value=False)
        self.entry(state_key="paid", state_value=True)
        self.entry("b", state_key="paid", state_value=True)
        self.assertEqual(self.service.detect_divergence("w", "a").conflicting_state_keys, [])

    def test_plan_distance_uses_persistent_node_ids(self):
        def plan(a, b, first="search"):
            return TaskPlan(nodes=[PlanNode(node_id=a, action=first), PlanNode(node_id=b, action="rank")],
                            edges=[PlanEdge(source=a, target=b)])
        self.assertEqual(normalized_plan_distance(plan("a", "b"), plan("a", "b")), 0)
        # One changed action at a matched node out of |V| + |E| = 3.
        self.assertAlmostEqual(normalized_plan_distance(plan("a", "b"), plan("a", "b", "book")), 1 / 3)
        # Fresh identifiers break the fixed correspondence (Appendix B.2).
        self.assertEqual(normalized_plan_distance(plan("a", "b"), plan("x", "y")), 1.0)

    def test_dependency_cycle_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "acyclic"):
            ProcedureGraph(steps=[ProcedureStep(step_id="a", instruction="A"), ProcedureStep(step_id="b", instruction="B")],
                           edges=[PlanEdge(source="a", target="b"), PlanEdge(source="b", target="a")])

    def test_delete_respects_strict_risk_threshold(self):
        decision = SafetyValidator(MemoryConfig(safety_risk_threshold=.25)).validate(
            SOPCandidate(operation=ProposalOperation.DELETE, target_sop_id="s", base_version=1,
                         source_workspace_ids=["w"]))
        self.assertFalse(decision.allowed)

    def test_confirmation_word_does_not_authorize_payment(self):
        for instruction in ("Pay without confirmation", "支付订单"):
            candidate = self.candidate(procedure=ProcedureGraph(steps=[ProcedureStep(instruction=instruction)]))
            self.assertFalse(SafetyValidator().validate(candidate).allowed)

    def test_general_sop_cannot_be_overwritten_by_context_variant(self):
        old = SOPVersion(sop_id="s", version=1, procedure=self.candidate().procedure,
                         metadata=SOPMetadata(title="Inspect", task_family="inspect"))
        update = self.candidate(operation=ProposalOperation.UPDATE, target_sop_id="s", base_version=1,
                                metadata=SOPMetadata(title="Inspect", task_family="inspect", context_conditions=["sandbox"]))
        self.assertFalse(SafetyValidator().validate(update, old).allowed)

    def test_variant_requires_verified_context_at_retrieval(self):
        c = self.candidate()
        sop = SOPVersion(sop_id="v", version=1, procedure=c.procedure,
                         metadata=SOPMetadata(title="Inspect", task_family="inspect", variant_of="parent", context_conditions=["sandbox"]))
        self.service.store.commit_sop(sop, None)
        self.assertEqual(self.service.retrieve_sops("Inspect"), [])
        self.assertEqual(len(self.service.retrieve_sops("Inspect", context_facts={"sandbox"})), 1)

    def test_failed_trial_counts_against_utility(self):
        c, _ = self.service.propose_sop(self.candidate())
        self.service.record_reproduction(self.trial(c, success=False, candidate_reward=-10))
        self.service.record_reproduction(self.trial(c))
        self.assertFalse(self.service.promote_candidate(c.candidate_id).promoted)

    def test_trial_cost_uses_configured_lambda(self):
        self.service.config = MemoryConfig(utility_cost_weight=0.1)
        c, _ = self.service.propose_sop(self.candidate())
        self.service.record_reproduction(self.trial(c, cost=2))
        self.assertTrue(self.service.promote_candidate(c.candidate_id).promoted)

    def test_commit_rolls_back_if_candidate_status_write_fails(self):
        c, _ = self.service.propose_sop(self.candidate())
        with patch.object(self.service.store, "_mark_promoted", side_effect=RuntimeError("write failed")):  # noqa: SIM117
            with self.assertRaisesRegex(RuntimeError, "write failed"):
                self.service.record_reproduction(self.trial(c))
        self.assertEqual(self.service.store.list_active_sops()[1], 0)
        self.assertNotEqual(self.service.store.get_candidate(c.candidate_id).status, CandidateStatus.PROMOTED)

    def test_concurrent_outcomes_do_not_lose_counts(self):
        c, _ = self.service.propose_sop(self.candidate())
        self.service.record_reproduction(self.trial(c))
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda _: self.service.record_sop_outcome(c.candidate_id, True), range(20)))
        self.assertEqual(self.service.store.get_sop(c.candidate_id).retrieval_count, 20)

    def test_agent_inference_is_not_verified_external_state(self):
        result = EvidenceResolver().resolve_world_state("paid", [Evidence(tier=EvidenceTier.AGENT_INFERENCE, source="a", value=True)])
        self.assertFalse(result.resolved)

    def test_zero_mean_net_benefit_passes_promotion(self):
        # Eq. 9 requires mean net benefit >= 0, not > 0.
        c, _ = self.service.propose_sop(self.candidate())
        self.service.record_reproduction(self.trial(c, candidate_reward=0))
        self.assertEqual(self.service.store.get_candidate(c.candidate_id).status, CandidateStatus.PROMOTED)

    def test_promotion_reports_exactly_the_eq9_gates(self):
        c, _ = self.service.propose_sop(self.candidate())
        decision = self.service.promote_candidate(c.candidate_id)
        self.assertFalse(decision.promoted)
        self.assertEqual(tuple(decision.gates), SEPMService.PROMOTION_GATES)
        self.assertFalse(decision.gates["net_benefit"])  # no recorded trials -> -inf

    def test_stale_update_is_superseded_not_raised(self):
        c, _ = self.service.propose_sop(self.candidate())
        self.service.record_reproduction(self.trial(c))
        updates = [
            self.service.propose_sop(self.candidate(
                operation=ProposalOperation.UPDATE, target_sop_id=c.candidate_id, base_version=1,
                procedure=ProcedureGraph(steps=[ProcedureStep(instruction=f"Inspect result {n}")])))[0]
            for n in (1, 2)
        ]
        self.service.record_reproduction(self.trial(updates[0]))
        self.service.record_reproduction(self.trial(updates[1]))
        self.assertEqual(self.service.store.get_sop(c.candidate_id).version, 2)
        self.assertEqual(self.service.store.get_candidate(updates[1].candidate_id).status, CandidateStatus.SUPERSEDED)

    def promoted_v1(self):
        c, _ = self.service.propose_sop(self.candidate())
        self.service.record_reproduction(self.trial(c))
        return c.candidate_id, self.service.store.get_sop(c.candidate_id)

    def test_concurrent_warnings_from_same_base_are_merged(self):
        sop_id, v1 = self.promoted_v1()
        a = ConditionalWarning(text="Found does not mean cleaned", trigger_condition="handoff", evidence=["w#1"])
        b = ConditionalWarning(text="Re-observe disputed locations", trigger_condition="conflict", evidence=["w#2"])
        with self.assertRaisesRegex(ValueError, "evidence"):
            self.service.attach_warnings(v1, [a.model_copy(update={"evidence": []})])
        v2 = self.service.attach_warnings(v1, [a])
        self.assertEqual((v2.version, v2.change.kind, v2.change.rebased_onto), (2, "warnings", None))
        # B learned its lesson on v1 too; the structure is unchanged, so it is appended to v2.
        v3 = self.service.attach_warnings(v1, [b])
        self.assertEqual((v3.version, v3.change.rebased_onto), (3, 2))
        self.assertEqual(self.service.store.get_sop(sop_id).metadata.warnings, [a, b])
        # A duplicate lesson creates no new version.
        self.assertEqual(self.service.attach_warnings(v1, [b]).version, 3)

    def test_only_head_and_previous_version_are_retained(self):
        sop_id, v1 = self.promoted_v1()
        for n in (1, 2):
            self.service.attach_warnings(v1, [ConditionalWarning(text=f"lesson {n}", evidence=["w"])])
        self.assertEqual(self.service.store.get_sop(sop_id, 2).version, 2)
        with self.assertRaises(NotFoundError):
            self.service.store.get_sop(sop_id, 1)

    def test_structural_change_blocks_stale_warnings(self):
        sop_id, v1 = self.promoted_v1()
        update, _ = self.service.propose_sop(self.candidate(
            operation=ProposalOperation.UPDATE, target_sop_id=sop_id, base_version=1,
            procedure=ProcedureGraph(steps=[ProcedureStep(instruction="Inspect result twice")])))
        self.service.record_reproduction(self.trial(update))
        with self.assertRaises(ConcurrencyError):
            self.service.attach_warnings(v1, [ConditionalWarning(text="lesson", evidence=["w"])])

    def test_structural_update_rebases_over_warning_only_changes(self):
        sop_id, v1 = self.promoted_v1()
        update, _ = self.service.propose_sop(self.candidate(
            operation=ProposalOperation.UPDATE, target_sop_id=sop_id, base_version=1,
            procedure=ProcedureGraph(steps=[ProcedureStep(instruction="Inspect result twice")])))
        lesson = ConditionalWarning(text="lesson", evidence=["w"])
        self.service.attach_warnings(v1, [lesson])
        self.service.record_reproduction(self.trial(update))
        head = self.service.store.get_sop(sop_id)
        self.assertEqual((head.version, head.change.kind, head.change.rebased_onto), (3, "structure", 2))
        self.assertEqual(head.procedure.steps[0].instruction, "Inspect result twice")
        self.assertEqual(head.metadata.warnings, [lesson])

    def test_legacy_plain_text_warning_is_coerced(self):
        metadata = SOPMetadata(title="Inspect", task_family="inspect", warnings=["Check twice"])
        self.assertEqual(metadata.warnings[0].text, "Check twice")

    def test_divergence_uses_assigned_goal_and_subgraph(self):
        plan = TaskPlan(
            nodes=[PlanNode(node_id="cups", action="clean cups"), PlanNode(node_id="dishes", action="clean dishes"),
                   PlanNode(node_id="store", action="store items")],
            edges=[PlanEdge(source="cups", target="store"), PlanEdge(source="dishes", target="store")],
        )
        self.service.create_workspace(Workspace(workspace_id="k", main_goal="Wash cups and dishes and store them",
                                                plan=plan))
        self.service.assign_agent("k", "b", goal="Clean the dirty dishes", node_ids=["dishes"])
        local = TaskPlan(nodes=[PlanNode(node_id="dishes", action="clean dishes")])
        self.service.append_blackboard(BlackboardEntry(
            workspace_id="k", agent_id="b", kind=BlackboardKind.PLAN, goal="Clean the dirty dishes", plan=local))
        report = self.service.detect_divergence("k", "b")
        self.assertEqual(report.plan_divergence, 0.0)
        self.assertFalse(report.requires_alignment)
        with self.assertRaisesRegex(ValueError, "unknown nodes"):
            self.service.assign_agent("k", "a", node_ids=["fridge"])

    def test_low_confidence_evidence_is_not_a_state_conflict(self):
        self.entry(state_key="paid", evidence=[Evidence(tier=EvidenceTier.TOOL_OBSERVATION, source="t", value=False)])
        self.entry("b", state_key="paid",
                   evidence=[Evidence(tier=EvidenceTier.TOOL_OBSERVATION, source="t", value=True, confidence=0.3)])
        self.assertEqual(self.service.detect_divergence("w", "a").conflicting_state_keys, [])

    def test_safety_matches_words_not_substrings(self):
        validator = SafetyValidator()
        for instruction, allowed in (("Select a size from the dropdown", True), ("Inspect the payload", True),
                                     ("Delete the draft", False), ("Dropped the old table", False)):
            candidate = self.candidate(procedure=ProcedureGraph(steps=[ProcedureStep(instruction=instruction)]))
            self.assertEqual(validator.validate(candidate).allowed, allowed, instruction)

    def test_prompt_keeps_late_confirmation_and_dependencies(self):
        steps = [ProcedureStep(step_id=str(i), instruction=f"Inspect {i}") for i in range(3)]
        steps.append(ProcedureStep(step_id="pay", instruction="Pay after confirmation", requires_confirmation=True))
        graph = ProcedureGraph(steps=steps, edges=[PlanEdge(source="2", target="pay")])
        text = SEPMBenchmarkRuntime._prompt_prefix([{"sop": {"procedure": graph.model_dump()}}])
        self.assertIn("Pay after confirmation", text)
        self.assertIn("Requires user confirmation", text)
        self.assertIn("2 -> pay", text)


if __name__ == "__main__":
    unittest.main()
