"""SEPM episode lifecycle: memory filling, blackboard alignment, and failure warnings."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from sepm.alignment import ActionRecord, derive_alignment
from sepm.config import MemoryConfig
from sepm.domains import ALFWORLD_MATCHER, ALFWorldDomain
from sepm.episode import ProceduralEpisode, string_list
from sepm.models import CandidateStatus, PlanEdge, PlanNode, TaskPlan
from sepm.service import SEPMService

GOAL = "Your task is to: put a clean mug in coffeemachine."
TAKE = ("take mug 1 from countertop 1", "You pick up the mug 1 from the countertop 1.")
CLEAN = ("clean mug 1 with sinkbasin 1", "You clean the mug 1 using the sinkbasin 1.")
PUT = ("put mug 1 in/on coffeemachine 1", "You put the mug 1 in/on the coffeemachine 1.")


class FakeSOPModel:
    """Returns a cited procedure for successes and a cited lesson for failures."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        self.prompts.append(user)
        if "FAILED" in user:
            return {"warnings": [{"text": "Clean the mug before putting it in the coffeemachine",
                                  "trigger_condition": "before placing a cleaned object",
                                  "evidence_steps": [2]}]}
        return {
            "title": "Clean and place a mug",
            "description": "Take, clean, then place the mug.",
            "applicability": "clean-then-place tasks",
            "atomic_steps": [
                {"instruction": "take mug", "evidence_steps": [1]},
                {"instruction": "clean mug with sinkbasin", "evidence_steps": [2]},
                {"instruction": "put mug in coffeemachine", "evidence_steps": [3]},
            ],
            "dependencies": [[1, 2], [2, 3]],
            "rationale": "observed in a successful episode",
        }


class EpisodeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.service = SEPMService(config=MemoryConfig(database_path=Path(self.tmp.name) / "memory.db"))
        self.model = FakeSOPModel()
        self.stats: dict[str, float] = {}
        self.count = 0

    def episode(self) -> ProceduralEpisode:
        self.count += 1
        episode = ProceduralEpisode(
            self.service, ALFWorldDomain(), workspace_id=f"ws-{self.count}", task_goal=GOAL,
            sop_model=self.model, stats=self.stats,
        )
        episode.retrieve(GOAL)
        return episode

    def run_steps(self, episode: ProceduralEpisode, *steps: tuple[str, str]) -> list[str]:
        notices: list[str] = []
        for action, observation in steps:
            notices += episode.record_step("solver", action, observation)
        return notices

    def test_memory_fills_through_trial_then_failure_attaches_warning(self) -> None:
        # 1. Unassisted success -> verified candidate, kept outside the active pool.
        first = self.episode()
        self.assertIsNone(first.context.reference)
        self.run_steps(first, TAKE, CLEAN, PUT)
        first.finish(success=True, reward=1.0)
        candidates, _ = self.service.store.list_candidates(CandidateStatus.VALIDATED, 10, 0)
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(len(candidate.procedure.edges), 2)
        self.assertEqual(self.service.retrieve_sops(GOAL, context_facts=first.context_facts), [])

        # 2. Same task type -> the candidate runs as a trial and Eq. 9 promotes it.
        second = self.episode()
        self.assertEqual(second.context.trial.sop_id, candidate.candidate_id)
        self.assertEqual(self.run_steps(second, TAKE, CLEAN, PUT), [])
        second.finish(success=True, reward=1.0)
        self.assertEqual(self.service.store.get_candidate(candidate.candidate_id).status, CandidateStatus.PROMOTED)
        trial = self.service.store.list_trials(candidate.candidate_id)[0]
        self.assertEqual(trial.baseline_reward, 1.0)

        # 3. Active SOP: skipping "clean" is detected from the blackboard, and the failure adds a warning.
        third = self.episode()
        self.assertEqual(third.context.active[0].sop_id, candidate.candidate_id)
        notices = self.run_steps(third, TAKE, PUT)
        self.assertTrue(any("prerequisite 'clean mug with sinkbasin'" in notice for notice in notices))
        self.assertEqual(third.take_notices(), notices)
        self.assertEqual(third.take_notices(), [])
        third.finish(success=False, reward=0.0)
        sop = self.service.store.get_sop(candidate.candidate_id)
        self.assertEqual(sop.version, 2)
        self.assertEqual(sop.metadata.warnings[0].evidence, ["ws-3#step-2"])
        self.assertEqual(sop.failure_count, 1)

    def test_failed_action_on_wrong_object_is_a_state_divergence(self) -> None:
        episode = self.episode()
        notices = self.run_steps(episode, TAKE, ("put apple 1 in/on fridge 1", "Nothing happens."))
        self.assertTrue(any("Verified state 'holding' is 'mug 1'" in notice for notice in notices))
        self.assertEqual(self.stats["alignment_world_state_conflict"], 1.0)

    def test_baseline_requires_unassisted_episodes(self) -> None:
        episode = self.episode()
        self.assertIsNone(episode.baseline_reward())

    def test_extraction_keeps_only_verified_steps_and_ordered_edges(self) -> None:
        episode = self.episode()
        self.run_steps(episode, TAKE, ("clean mug 1 with sinkbasin 1", "Nothing happens."), CLEAN)
        candidate = episode.candidate_from_json(
            {
                "title": "t",
                "atomic_steps": [
                    {"instruction": "take mug", "evidence_steps": [1]},
                    {"instruction": "heat mug", "evidence_steps": []},
                    {"instruction": "clean mug", "evidence_steps": [2]},
                    {"instruction": "clean mug again", "evidence_steps": [3]},
                ],
                # [4, 1] contradicts the observed order and is dropped.
                "dependencies": [[1, 4], [4, 1], [2, 4]],
            },
            reward=1.0,
        )
        self.assertEqual([step.instruction for step in candidate.procedure.steps], ["take mug", "clean mug again"])
        self.assertEqual([(e.source, e.target) for e in candidate.procedure.edges], [("step-1", "step-2")])
        self.assertAlmostEqual(candidate.causal_confidence, 0.5)

    def test_context_facts_follow_alfworld_task_type(self) -> None:
        domain = ALFWorldDomain()
        self.assertEqual(domain.context_facts("heat some egg and put it in garbagecan"),
                         {"alfworld_task_type:pick_heat_then_place_in_recep"})
        self.assertEqual(domain.context_facts("put two cellphone in sofa"),
                         {"alfworld_task_type:pick_two_obj_and_place"})

    def test_string_list_preserves_scalar_as_one_item(self) -> None:
        self.assertEqual(string_list("use valid actions", "applicability"), ["use valid actions"])
        self.assertEqual(string_list(["one", 2], "exclusions"), ["one", "2"])
        with self.assertRaises(TypeError):
            string_list({"bad": "shape"}, "applicability")


class AlignmentTests(unittest.TestCase):
    PLAN = TaskPlan(
        nodes=[PlanNode(node_id="go", action="go to the sinkbasin"), PlanNode(node_id="take", action="take mug"),
               PlanNode(node_id="clean", action="clean mug with sinkbasin"),
               PlanNode(node_id="put", action="put mug in coffeemachine")],
        edges=[PlanEdge(source="take", target="go"), PlanEdge(source="go", target="clean"),
               PlanEdge(source="clean", target="put")],
    )

    def test_navigation_steps_are_contracted_and_unreached_steps_ignored(self) -> None:
        view = derive_alignment(self.PLAN, [ActionRecord(*TAKE)], ALFWORLD_MATCHER)
        self.assertEqual([node.node_id for node in view.reference_view.nodes], ["take"])
        self.assertEqual(view.local_plan, view.reference_view)

    def test_skipped_prerequisite_clears_after_correction(self) -> None:
        skipped = derive_alignment(self.PLAN, [ActionRecord(*TAKE), ActionRecord(*PUT)], ALFWORLD_MATCHER)
        self.assertEqual(skipped.missing_prerequisites, [("clean", "put")])
        fixed = derive_alignment(
            self.PLAN, [ActionRecord(*TAKE), ActionRecord(*PUT), ActionRecord(*CLEAN), ActionRecord(*PUT)],
            ALFWORLD_MATCHER,
        )
        self.assertEqual(fixed.missing_prerequisites, [])

    def test_off_plan_action_counts_until_verified_progress(self) -> None:
        records = [ActionRecord(*TAKE), ActionRecord("put apple 1 in/on fridge 1", "You put the apple 1 in/on the fridge 1.")]
        self.assertEqual(derive_alignment(self.PLAN, records, ALFWORLD_MATCHER).off_plan_actions,
                         ["put apple 1 in/on fridge 1"])
        records.append(ActionRecord(*CLEAN))
        self.assertEqual(derive_alignment(self.PLAN, records, ALFWORLD_MATCHER).off_plan_actions, [])


if __name__ == "__main__":
    unittest.main()
