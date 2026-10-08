"""Adapt the core service to common agent-framework lifecycles.

This module does not depend on LangGraph, AutoGen, or CrewAI. It defines three
lightweight hooks for agent startup, agent steps, and SOP completion. Framework
integrations call them from the appropriate callbacks. Every return value is a
JSON-compatible dictionary for cross-process transfer or framework state.
"""

from __future__ import annotations

from typing import Any

from .models import AgentProfile, BlackboardEntry, TaskPlan, Workspace
from .service import SEPMService


class AgentMemoryAdapter:
    """Thin framework adapter; ``SEPMService`` retains all business rules."""

    def __init__(self, service: SEPMService) -> None:
        self.service = service

    def on_agent_start(
        self,
        profile: AgentProfile,
        workspace: Workspace,
        task_query: str,
        query_plan: TaskPlan | None = None,
        sop_limit: int = 5,
    ) -> dict[str, Any]:
        """Register the agent/workspace and inject relevant procedural memory."""
        # Agent and workspace writes are upserts, so retried startup hooks create no duplicates.
        self.service.register_agent(profile)
        workspace = self.service.create_workspace(workspace)
        sops = self.service.retrieve_sops(task_query, sop_limit, query_plan)
        return {
            "private_memory": profile.model_dump(mode="json"),
            "shared_task_state": workspace.model_dump(mode="json"),
            "procedural_memory": [item.model_dump(mode="json") for item in sops],
        }

    def on_agent_step(self, entry: BlackboardEntry) -> dict[str, Any]:
        """Store one auditable step and return the agent's latest divergence report."""
        stored = self.service.append_blackboard(entry)
        divergence = self.service.detect_divergence(entry.workspace_id, entry.agent_id)
        return {
            "entry": stored.model_dump(mode="json"),
            "divergence": divergence.model_dump(mode="json"),
        }

    def on_sop_outcome(self, sop_id: str, success: bool, query: str) -> dict[str, Any]:
        """Feed the actual SOP outcome into success statistics and retrieval routing."""
        return self.service.record_sop_outcome(sop_id, success, query=query).model_dump(mode="json")
