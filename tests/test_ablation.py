"""最小组件条件必须真正打开/关闭对应机制。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from team_memory.ablation.runtime import apply_ablation
from team_memory.config import MemoryConfig
from team_memory.models import (
    AgentProfile,
    BlackboardEntry,
    BlackboardKind,
    ProcedureGraph,
    ProcedureStep,
    ProposalOperation,
    SOPCandidate,
    SOPMetadata,
    Workspace,
)
from team_memory.service import TeamMemoryService


class AblationTests(unittest.TestCase):
    def test_component_conditions_map_to_expected_switches(self) -> None:
        base = MemoryConfig()
        no_extra = apply_ablation(base, "no-extra-components")
        blackboard_only = apply_ablation(base, "blackboard-only")
        sop_only = apply_ablation(base, "sop-only")
        divergence_only = apply_ablation(base, "divergence-only")

        self.assertFalse(no_extra.enable_procedural_memory)
        self.assertFalse(no_extra.enable_divergence_detection)
        self.assertFalse(no_extra.share_blackboard_across_agents)

        self.assertFalse(blackboard_only.enable_procedural_memory)
        self.assertFalse(blackboard_only.enable_divergence_detection)
        self.assertTrue(blackboard_only.share_blackboard_across_agents)

        self.assertTrue(sop_only.enable_procedural_memory)
        self.assertFalse(sop_only.enable_divergence_detection)

        self.assertFalse(divergence_only.enable_procedural_memory)
        self.assertTrue(divergence_only.enable_divergence_detection)

    def test_sop_only_keeps_sop_path_but_disables_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = apply_ablation(
                MemoryConfig(database_path=Path(directory) / "sop.db"),
                "sop-only",
            )
            service = TeamMemoryService(config=config)
            service.register_agent(AgentProfile(agent_id="worker", role="executor"))
            service.create_workspace(Workspace(workspace_id="w", main_goal="safe goal"))
            service.append_blackboard(
                BlackboardEntry(
                    workspace_id="w",
                    agent_id="worker",
                    kind=BlackboardKind.TASK,
                    task="opposite goal",
                    goal="opposite goal",
                )
            )
            self.assertFalse(service.detect_divergence("w", "worker").requires_alignment)
            candidate, decision = service.propose_sop(
                SOPCandidate(
                    operation=ProposalOperation.CREATE,
                    procedure=ProcedureGraph(steps=[ProcedureStep(instruction="Verify result")]),
                    metadata=SOPMetadata(title="verify", task_family="test"),
                    source_workspace_ids=["w"],
                )
            )
            self.assertTrue(decision.allowed)
            self.assertEqual(candidate.status.value, "pending")

    def test_divergence_only_keeps_alignment_but_blocks_sop_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = apply_ablation(
                MemoryConfig(database_path=Path(directory) / "divergence.db"),
                "divergence-only",
            )
            service = TeamMemoryService(config=config)
            service.register_agent(AgentProfile(agent_id="worker", role="executor"))
            service.create_workspace(Workspace(workspace_id="w", main_goal="safe goal"))
            service.append_blackboard(
                BlackboardEntry(
                    workspace_id="w",
                    agent_id="worker",
                    kind=BlackboardKind.TASK,
                    task="opposite goal",
                    goal="opposite goal",
                )
            )
            self.assertTrue(service.detect_divergence("w", "worker").requires_alignment)
            candidate, decision = service.propose_sop(
                SOPCandidate(
                    operation=ProposalOperation.CREATE,
                    procedure=ProcedureGraph(steps=[ProcedureStep(instruction="Verify result")]),
                    metadata=SOPMetadata(title="verify", task_family="test"),
                    source_workspace_ids=["w"],
                )
            )
            self.assertFalse(decision.allowed)
            self.assertEqual(candidate.status.value, "rejected")
            self.assertEqual(service.retrieve_sops("verify result"), [])

    def test_no_extra_components_disables_both_added_mechanisms(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = apply_ablation(
                MemoryConfig(database_path=Path(directory) / "none.db"),
                "no-extra-components",
            )
            service = TeamMemoryService(config=config)
            service.register_agent(AgentProfile(agent_id="worker", role="executor"))
            service.create_workspace(Workspace(workspace_id="w", main_goal="safe goal"))
            service.append_blackboard(
                BlackboardEntry(
                    workspace_id="w",
                    agent_id="worker",
                    kind=BlackboardKind.TASK,
                    task="opposite goal",
                    goal="opposite goal",
                )
            )
            self.assertFalse(service.detect_divergence("w", "worker").requires_alignment)
            self.assertEqual(service.retrieve_sops("safe goal"), [])
            with self.assertRaises(ValueError):
                service.list_blackboard("w")


if __name__ == "__main__":
    unittest.main()
