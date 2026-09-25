"""最小端到端示例：候选从提出到首个 trial 成功后自动发布并被检索。

脚本会在当前目录写 ``demo_memory.db``，适合人工观察 API 流程；重复运行会保留历史
SOP。自动化测试请使用 ``tests/test_core.py`` 中的临时数据库，避免状态互相影响。
"""

from __future__ import annotations

from pathlib import Path

from team_memory.models import (
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
from team_memory.service import TeamMemoryService


def main() -> None:
    """构造两个工作区、提交一个 SOP 候选、验证并展示检索结果。"""
    database = Path("demo_memory.db")
    service = TeamMemoryService(database)
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
    # record_reproduction 在首个合格成功 trial 返回前已自动完成 SOP 写入。
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
    # 显式 promote 仍可作为幂等检查/重试入口，不会重复创建版本。
    promotion = service.promote_candidate(candidate.candidate_id)
    print("promoted:", promotion.promoted, "sop:", promotion.sop.sop_id if promotion.sop else None)
    for result in service.retrieve_sops("rank verified catalog results"):
        print(result.sop.metadata.title, round(result.score, 3), result.features)


if __name__ == "__main__":
    main()
