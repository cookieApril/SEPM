"""Minimal end-to-end example from proposal to promotion and retrieval.

The script writes ``sepm_demo.db`` in the current directory for manual API
inspection. Repeated runs preserve SOP history. Automated tests should use the
temporary databases in ``tests/test_core.py`` to keep state isolated.
"""

from __future__ import annotations

from pathlib import Path

from sepm.models import (
    AgentProfile,
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
from sepm.service import SEPMService


def main() -> None:
    """Create two workspaces, validate an SOP candidate, and show retrieval results."""
    database = Path("sepm_demo.db")
    service = SEPMService(database)
    service.register_agent(AgentProfile(agent_id="researcher", role="find verified evidence"))
    for workspace_id, action in (("task-a", "search catalog"), ("task-b", "lookup index")):
        service.create_workspace(
            Workspace(
                workspace_id=workspace_id,
                main_goal="retrieve and verify candidates",
                plan=TaskPlan(nodes=[PlanNode(node_id="query", action=action)]),
            )
        )
    verify = ProcedureStep(instruction="Query an authoritative source", action_type="query")
    rank = ProcedureStep(instruction="Rank only verified candidates", action_type="rank")
    candidate = SOPCandidate(
        operation=ProposalOperation.CREATE,
        procedure=ProcedureGraph(
            steps=[verify, rank],
            edges=[PlanEdge(source=verify.step_id, target=rank.step_id)],
        ),
        metadata=SOPMetadata(
            title="Evidence-first retrieval",
            task_family="retrieval",
            applicability=["catalog search", "document lookup"],
        ),
        source_workspace_ids=["task-a", "task-b"],
        state_verified=True,
        causal_confidence=0.9,
    )
    candidate, safety = service.propose_sop(candidate)
    print("safety:", safety.allowed)
    # record_reproduction writes the SOP before the first qualified trial returns.
    service.record_reproduction(
        ReproductionTrial(
            candidate_id=candidate.candidate_id,
            workspace_id="task-a",
            task_family="catalog",
            environment_fingerprint="catalog-v1",
            success=True,
            baseline_reward=0.4,
            candidate_reward=0.9,
            cost=0.1,
            state_verified=True,
            causal_supported=True,
        )
    )
    # Explicit promotion remains an idempotent check/retry path and creates no duplicate version.
    promotion = service.promote_candidate(candidate.candidate_id)
    print("promoted:", promotion.promoted, "sop:", promotion.sop.sop_id if promotion.sop else None)
    for result in service.retrieve_sops("rank verified catalog results"):
        print(result.sop.metadata.title, round(result.score, 3), result.features)


if __name__ == "__main__":
    main()
