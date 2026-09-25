"""把核心服务适配到常见 Agent 框架的生命周期。

该模块不依赖 LangGraph、AutoGen 或 CrewAI 本身，只定义三个轻量 hook：Agent
启动、Agent 执行一步、SOP 执行结束。框架集成层负责在合适的回调中调用它们。
所有返回值都转换为 JSON 兼容字典，便于跨进程或写入框架状态。
"""

from __future__ import annotations

from typing import Any

from .models import AgentProfile, BlackboardEntry, TaskPlan, Workspace
from .service import TeamMemoryService


class AgentMemoryAdapter:
    """面向框架的薄适配器；业务规则仍全部由 ``TeamMemoryService`` 执行。"""

    def __init__(self, service: TeamMemoryService) -> None:
        self.service = service

    def on_agent_start(
        self,
        profile: AgentProfile,
        workspace: Workspace,
        task_query: str,
        query_plan: TaskPlan | None = None,
        sop_limit: int = 5,
    ) -> dict[str, Any]:
        """注册 Agent/工作区，并在任务开始前注入最相关的程序性记忆。"""
        # 注册和工作区写入均是 upsert，因此框架重试启动 hook 不会制造重复记录。
        self.service.register_agent(profile)
        workspace = self.service.create_workspace(workspace)
        sops = self.service.retrieve_sops(task_query, sop_limit, query_plan)
        return {
            "private_memory": profile.model_dump(mode="json"),
            "shared_task_state": workspace.model_dump(mode="json"),
            "procedural_memory": [item.model_dump(mode="json") for item in sops],
        }

    def on_agent_step(self, entry: BlackboardEntry) -> dict[str, Any]:
        """保存一步可审计轨迹，并立即返回该 Agent 的最新偏差报告。"""
        stored = self.service.append_blackboard(entry)
        divergence = self.service.detect_divergence(entry.workspace_id, entry.agent_id)
        return {
            "entry": stored.model_dump(mode="json"),
            "divergence": divergence.model_dump(mode="json"),
        }

    def on_sop_outcome(self, sop_id: str, success: bool, query: str) -> dict[str, Any]:
        """把 SOP 的真实执行结果反馈给成功率统计和检索路由器。"""
        return self.service.record_sop_outcome(sop_id, success, query=query).model_dump(mode="json")
