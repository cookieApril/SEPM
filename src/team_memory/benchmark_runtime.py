"""Runtime hooks for injecting Team Memory into official benchmark loops.

Official benchmarks should keep their own environments, agents, and evaluators.
This module only supplies the shared Team Memory lifecycle:

``start_agent -> observe -> before_action -> after_action -> finish_case``.

Adapters for WebArena, MARBLE/MultiAgentBench, OfficeBench, AgentNet, and
ALFWorld can call these hooks inside the official action loop without copying
benchmark logic or fabricating scores.
"""

from __future__ import annotations

import os
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .ablation.runtime import config_from_environment
from .adapter import AgentMemoryAdapter
from .config import MemoryConfig
from .evaluation_adapter import EvaluationContext
from .models import (
    AgentProfile,
    BlackboardEntry,
    BlackboardKind,
    Evidence,
    EvidenceTier,
    PlanEdge,
    PlanNode,
    TaskPlan,
    Workspace,
)
from .service import TeamMemoryService
from .integrations import GMemoryBridge, HostRuntimeBridge, GMemorySnapshotError
from .integrations.gmemory_bridge import ensure_development_snapshot


def _text(value: Any, limit: int = 8_000) -> str:
    """Convert benchmark-native objects to bounded blackboard strings."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:limit]
    return repr(value)[:limit]


def _workspace_id(context: EvaluationContext) -> str:
    parts = [
        context.benchmark,
        context.task,
        context.mas_framework,
        context.memory_method,
        context.ablation,
        f"s{context.seed}",
        context.case_id or "case",
    ]
    raw = "__".join(part.replace("/", "-").replace(" ", "_") for part in parts)
    if len(raw) <= 128:
        return raw
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{raw[:108]}__{digest}"


def _database_path(context: EvaluationContext) -> Path:
    configured = os.environ.get("TEAM_MEMORY_DB")
    if configured:
        return Path(configured)
    return context.output_path.with_suffix(".team-memory.db")


def _plan_from_edges(actions: list[str], edges: list[tuple[int, int]] | None = None) -> TaskPlan:
    nodes = [
        PlanNode(node_id=f"n{index}", action=_text(action or f"step {index}", 1_000))
        for index, action in enumerate(actions)
    ]
    plan_edges = [
        PlanEdge(source=f"n{source}", target=f"n{target}")
        for source, target in (edges or [])
    ]
    return TaskPlan(nodes=nodes, edges=plan_edges)


@dataclass
class TeamMemoryBenchmarkRuntime:
    """Small integration surface for third-party benchmark adapters.

    The runtime is intentionally stateful only within one benchmark case. The
    durable state remains in ``TeamMemoryService``/SQLite, and the outer
    evaluation runner still owns the seed+case_id checkpoint.
    """

    context: EvaluationContext
    service: TeamMemoryService
    adapter: AgentMemoryAdapter
    workspace: Workspace
    config: MemoryConfig
    enabled: bool
    retrieved_sop_ids: set[str] = field(default_factory=set)
    used_sop_ids: set[str] = field(default_factory=set)
    sop_outcomes: dict[str, bool] = field(default_factory=dict)
    divergence_events: int = 0
    recovered_divergence_events: int = 0
    divergence_reason_counts: dict[str, int] = field(default_factory=dict)
    unsafe_accepted_count: int = 0
    blackboard_entries: int = 0
    duplicate_divergence_events: int = 0
    recovery_proposed_events: int = 0
    recovery_executed_events: int = 0
    recovery_verified_events: int = 0
    recovery_verified_success_events: int = 0
    last_divergence_signature: dict[str, str] = field(default_factory=dict)
    active_divergence_events: dict[str, str] = field(default_factory=dict)
    divergence_lifecycle: list[dict[str, Any]] = field(default_factory=list)
    external_memory_bridge: GMemoryBridge | None = None
    host_bridge: HostRuntimeBridge | None = None
    host_trace: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_environment(
        cls,
        *,
        main_goal: str,
        plan_actions: list[str] | None = None,
        plan_edges: list[tuple[int, int]] | None = None,
        enabled_methods: set[str] | None = None,
    ) -> "TeamMemoryBenchmarkRuntime":
        context = EvaluationContext.from_environment()
        if context.memory_method == "gmemory":
            raise GMemorySnapshotError(
                "generic runtime has no upstream G-Memory execution; use the official ALFWorld delegate"
            )
        if context.mas_framework == "dylan":
            raise RuntimeError(
                "generic runtime has no upstream DyLAN routing; use the official ALFWorld delegate"
            )
        base = MemoryConfig(database_path=_database_path(context))
        config = config_from_environment(base)
        method_enabled = context.memory_method in (enabled_methods or {"team-memory", "gmemory"})
        has_active_component = (
            config.enable_procedural_memory
            or config.enable_divergence_detection
            or config.share_blackboard_across_agents
        )
        external_memory_bridge = None
        if context.memory_method == "gmemory":
            snapshot = os.environ.get("TEAM_MEMORY_GMEMORY_SNAPSHOT")
            project_root = Path(os.environ.get("TEAM_MEMORY_PROJECT_ROOT", ".")).resolve()
            snapshot_path = (
                Path(snapshot)
                if snapshot
                else ensure_development_snapshot(
                    project_root=project_root,
                    benchmark_task=context.task,
                    evaluation_case_id=context.case_id,
                )
            )
            external_memory_bridge = GMemoryBridge.open_snapshot(snapshot_path, read_only=True)
            evidence_path = os.environ.get("TEAM_MEMORY_GMEMORY_EVIDENCE")
            if evidence_path:
                external_memory_bridge.evidence_path = Path(evidence_path)
        enabled = method_enabled and (
            has_active_component or external_memory_bridge is not None
        )
        host_bridge = HostRuntimeBridge.build(
            mas=context.mas_framework,
            task=context.task,
            project_root=Path(os.environ.get("TEAM_MEMORY_PROJECT_ROOT", ".")).resolve(),
        )
        service = TeamMemoryService(config=config)
        workspace = Workspace(
            workspace_id=_workspace_id(context),
            main_goal=main_goal,
            plan=_plan_from_edges(plan_actions or [main_goal], plan_edges),
        )
        return cls(
            context=context,
            service=service,
            adapter=AgentMemoryAdapter(service),
            workspace=workspace,
            config=config,
            enabled=enabled,
            external_memory_bridge=external_memory_bridge,
            host_bridge=host_bridge,
        )

    def start_agent(
        self,
        agent_id: str,
        role: str,
        *,
        current_task: str,
        task_query: str | None = None,
        sop_limit: int = 5,
    ) -> dict[str, Any]:
        """Register an agent and return SOP context to inject into its prompt/state."""
        host_prefix = (
            self.host_bridge.actor_prefix(agent_id=agent_id, observation=_text(task_query or current_task))
            if self.host_bridge is not None
            else ""
        )
        if not self.enabled:
            return {
                "enabled": False,
                "prompt_prefix": host_prefix,
                "procedural_memory": [],
            }
        if self.external_memory_bridge is not None:
            profile = AgentProfile(
                agent_id=agent_id,
                role=_text(role, 512),
                current_task=_text(current_task, 4_000),
                permanent_facts={
                    "benchmark": self.context.benchmark,
                    "task": self.context.task,
                    "case_id": self.context.case_id,
                    "mas": self.context.mas_framework,
                },
            )
            self.service.register_agent(profile)
            self.service.create_workspace(self.workspace)
            retrievals = self.external_memory_bridge.retrieve(
                task=_text(task_query or current_task or self.workspace.main_goal, 4_000),
                observation=_text(current_task, 4_000),
                agent_role=_text(role, 512),
                top_k=sop_limit,
            )
            prefix = "\n\n".join(
                item
                for item in (
                    host_prefix,
                    self.external_memory_bridge.format_for_actor(retrievals),
                )
                if item
            )
            return {
                "enabled": True,
                "prompt_prefix": prefix,
                "procedural_memory": [],
                "gmemory_retrievals": [
                    {
                        "node_id": item.node_id,
                        "score": item.score,
                        "task": item.task,
                    }
                    for item in retrievals
                ],
            }
        profile = AgentProfile(
            agent_id=agent_id,
            role=_text(role, 512),
            current_task=_text(current_task, 4_000),
            permanent_facts={
                "benchmark": self.context.benchmark,
                "task": self.context.task,
                "case_id": self.context.case_id,
                "mas": self.context.mas_framework,
            },
        )
        payload = self.adapter.on_agent_start(
            profile,
            self.workspace,
            _text(task_query or current_task or self.workspace.main_goal, 4_000),
            self.workspace.plan,
            sop_limit=sop_limit,
        )
        sops = payload["procedural_memory"]
        for item in sops:
            sop = item.get("sop", {})
            if sop.get("sop_id"):
                self.retrieved_sop_ids.add(str(sop["sop_id"]))
        return {
            **payload,
            "enabled": True,
            "prompt_prefix": "\n\n".join(
                item for item in (host_prefix, self._prompt_prefix(sops)) if item
            ),
        }

    def observe(
        self,
        agent_id: str,
        observation: Any,
        *,
        task: str | None = None,
        state_key: str | None = None,
        state_value: Any = None,
        evidence_tier: EvidenceTier = EvidenceTier.TOOL_OBSERVATION,
        evidence_source: str = "official-benchmark-observation",
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        evidence: list[Evidence] = []
        if state_key is not None:
            evidence.append(
                Evidence(
                    tier=evidence_tier,
                    source=evidence_source,
                    value=state_value,
                    confidence=1.0,
                )
            )
        return self._append(
            BlackboardEntry(
                workspace_id=self.workspace.workspace_id,
                agent_id=agent_id,
                kind=BlackboardKind.OBSERVATION,
                task=_text(task, 4_000) if task is not None else None,
                observation=_text(observation),
                state_key=state_key,
                state_value=state_value,
                evidence=evidence,
            )
        )

    def before_action(
        self,
        agent_id: str,
        action: Any,
        *,
        task: str | None = None,
        goal: str | None = None,
        plan: TaskPlan | None = None,
        used_sop_id: str | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "requires_alignment": False}
        if used_sop_id:
            self.used_sop_ids.add(used_sop_id)
        return self._append(
            BlackboardEntry(
                workspace_id=self.workspace.workspace_id,
                agent_id=agent_id,
                kind=BlackboardKind.ACTION,
                task=_text(task, 4_000) if task is not None else None,
                action=_text(action, 4_000),
                goal=_text(goal, 10_000) if goal is not None else None,
                plan=plan,
            )
        )

    def after_action(
        self,
        agent_id: str,
        result: Any,
        *,
        task: str | None = None,
        state_key: str | None = None,
        state_value: Any = None,
        authoritative: bool = False,
        unsafe_accepted: bool = False,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        if unsafe_accepted:
            self.unsafe_accepted_count += 1
        tier = EvidenceTier.AUTHORITATIVE_STATE if authoritative else EvidenceTier.TOOL_OBSERVATION
        evidence = []
        if state_key is not None:
            evidence.append(
                Evidence(
                    tier=tier,
                    source="official-benchmark-result",
                    value=state_value,
                    confidence=1.0,
                )
            )
        return self._append(
            BlackboardEntry(
                workspace_id=self.workspace.workspace_id,
                agent_id=agent_id,
                kind=BlackboardKind.RESULT,
                task=_text(task, 4_000) if task is not None else None,
                result=_text(result),
                state_key=state_key,
                state_value=state_value,
                evidence=evidence,
            )
        )

    def record_error(self, agent_id: str, error: Any, *, task: str | None = None) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        return self._append(
            BlackboardEntry(
                workspace_id=self.workspace.workspace_id,
                agent_id=agent_id,
                kind=BlackboardKind.ERROR,
                task=_text(task, 4_000) if task is not None else None,
                error=_text(error),
            )
        )

    def record_recovery_proposed(
        self,
        agent_id: str,
        proposal: Any,
        *,
        target: str | None = None,
        event_id: str | None = None,
    ) -> dict[str, Any]:
        """Record a proposed repair for an already detected divergence event."""
        if not self.enabled:
            return {"enabled": False}
        event_id = event_id or self.active_divergence_events.get(agent_id)
        if not event_id:
            return {"enabled": True, "recorded": False, "reason": "no_active_conflict"}
        if event_id != self.active_divergence_events.get(agent_id):
            return {"enabled": True, "recorded": False, "reason": "stale_or_foreign_conflict"}
        self.recovery_proposed_events += 1
        event = {
            "event_id": event_id,
            "stage": "recovery_proposed",
            "agent_id": agent_id,
            "target": _text(target, 1_000) if target is not None else "",
            "proposal": _text(proposal, 2_000),
        }
        self.divergence_lifecycle.append(event)
        return {"enabled": True, "recorded": True, **event}

    def record_recovery_executed(
        self,
        agent_id: str,
        action: Any,
        *,
        event_id: str | None = None,
    ) -> dict[str, Any]:
        """Record that a repair was actually executed in the host loop."""
        if not self.enabled:
            return {"enabled": False}
        event_id = event_id or self.active_divergence_events.get(agent_id)
        if not event_id:
            return {"enabled": True, "recorded": False, "reason": "no_active_conflict"}
        if event_id != self.active_divergence_events.get(agent_id) or not any(
            event["event_id"] == event_id and event["stage"] == "recovery_proposed"
            for event in self.divergence_lifecycle
        ):
            return {"enabled": True, "recorded": False, "reason": "repair_not_proposed"}
        self.recovery_executed_events += 1
        event = {
            "event_id": event_id,
            "stage": "recovery_executed",
            "agent_id": agent_id,
            "action": _text(action, 2_000),
        }
        self.divergence_lifecycle.append(event)
        return {"enabled": True, "recorded": True, **event}

    def record_recovery_verified(
        self,
        agent_id: str,
        evidence: Any,
        *,
        success: bool,
        event_id: str | None = None,
    ) -> dict[str, Any]:
        """Record authoritative post-repair evidence and success/failure."""
        if not self.enabled:
            return {"enabled": False}
        event_id = event_id or self.active_divergence_events.get(agent_id)
        if not event_id:
            return {"enabled": True, "recorded": False, "reason": "no_active_conflict"}
        if event_id != self.active_divergence_events.get(agent_id) or not any(
            event["event_id"] == event_id and event["stage"] == "recovery_executed"
            for event in self.divergence_lifecycle
        ) or not _text(evidence).strip():
            return {"enabled": True, "recorded": False, "reason": "repair_not_executed_or_no_evidence"}
        self.recovery_verified_events += 1
        if success:
            self.recovered_divergence_events += 1
            self.recovery_verified_success_events += 1
            self.active_divergence_events.pop(agent_id, None)
            self.last_divergence_signature.pop(agent_id, None)
        event = {
            "event_id": event_id,
            "stage": "recovery_verified",
            "agent_id": agent_id,
            "success": bool(success),
            "evidence": _text(evidence, 2_000),
        }
        self.divergence_lifecycle.append(event)
        return {"enabled": True, "recorded": True, **event}

    def record_sop_outcome(self, sop_id: str, *, success: bool) -> dict[str, Any]:
        """Record a verified outcome for an explicitly adopted SOP, once per episode."""
        if sop_id not in self.used_sop_ids:
            raise ValueError("SOP outcome requires explicit adoption")
        if sop_id in self.sop_outcomes:
            if self.sop_outcomes[sop_id] != success:
                raise ValueError("SOP outcome already recorded with a different value")
            return {"recorded": False, "reason": "already_recorded"}
        result = self.adapter.on_sop_outcome(sop_id, success, self.workspace.main_goal)
        self.sop_outcomes[sop_id] = success
        return {"recorded": True, "sop": result}

    def finish_metrics(self, official_metrics: dict[str, float] | None = None) -> dict[str, float]:
        """Return Team Memory metrics to merge into the official result schema."""
        metrics = dict(official_metrics or {})
        metrics.update(
            {
                "team_memory_enabled": 1.0 if self.enabled else 0.0,
                "procedural_memory_enabled": (
                    1.0 if self.enabled and self.config.enable_procedural_memory else 0.0
                ),
                "divergence_enabled": (
                    1.0 if self.enabled and self.config.enable_divergence_detection else 0.0
                ),
                "blackboard_enabled": (
                    1.0 if self.enabled and self.config.share_blackboard_across_agents else 0.0
                ),
                "blackboard_entries": float(self.blackboard_entries),
                "sop_retrieval_count": float(len(self.retrieved_sop_ids)),
                "sop_adoption_count": float(len(self.used_sop_ids)),
                "sop_verified_outcome_count": float(len(self.sop_outcomes)),
                "sop_reuse_success": float(any(self.sop_outcomes.values())),
                "divergence_events": float(self.divergence_events),
                "conflict_detected_count": float(self.divergence_events),
                "duplicate_conflict_count": float(self.duplicate_divergence_events),
                "recovery_proposed_count": float(self.recovery_proposed_events),
                "recovery_executed_count": float(self.recovery_executed_events),
                "recovery_verified_count": float(self.recovery_verified_events),
                "recovery_verified_success_count": float(
                    self.recovery_verified_success_events
                ),
                "goal_divergence_count": float(
                    self.divergence_reason_counts.get("goal_divergence", 0)
                ),
                "plan_divergence_count": float(
                    self.divergence_reason_counts.get("plan_divergence", 0)
                ),
                "world_state_conflict_count": float(
                    self.divergence_reason_counts.get("world_state_conflict", 0)
                ),
                "divergence_recovery": (
                    self.recovered_divergence_events / self.divergence_events
                    if self.divergence_events
                    else 0.0
                ),
                "unsafe_accepted": float(self.unsafe_accepted_count),
                "gmemory_retrieval_invocations": float(
                    self.external_memory_bridge.retrieval_invocations
                    if self.external_memory_bridge is not None
                    else 0
                ),
                "gmemory_retrieval_count": float(
                    self.external_memory_bridge.retrieval_count
                    if self.external_memory_bridge is not None
                    else 0
                ),
                "host_trace_events": float(len(self.host_trace)),
            }
        )
        if self.context.memory_method == "gmemory" and self.external_memory_bridge is not None:
            self.external_memory_bridge.write_evidence()
        self._write_host_trace()
        return metrics

    def identity_metadata(self) -> dict[str, Any]:
        memory_identity = "none"
        memory_mode = f"{self.context.task}:{self.context.memory_method}:official-loop"
        snapshot_id = ""
        if self.context.memory_method == "gmemory":
            if self.external_memory_bridge is None:
                raise GMemorySnapshotError("G-Memory result cannot be accepted without bridge invocation")
            memory_identity = self.external_memory_bridge.identity
            memory_mode = f"{self.context.task}:gmemory:read-only-frozen-snapshot"
            snapshot_id = self.external_memory_bridge.snapshot_id
        elif self.context.memory_method == "team-memory":
            memory_identity = "team_memory.benchmark_runtime.TeamMemoryBenchmarkRuntime:gems-runtime-hooks"
            memory_mode = f"{self.context.task}:team-memory:gems-runtime-hooks"
        metadata = {
            "memory_execution_mode": memory_mode,
            "memory_adapter_identity": memory_identity,
            "memory_adapter_class": (
                self.external_memory_bridge.__class__.__name__
                if self.external_memory_bridge is not None
                else ("TeamMemoryBenchmarkRuntime" if self.context.memory_method == "team-memory" else "NoAddedMemory")
            ),
            "memory_adapter_module": (
                self.external_memory_bridge.__class__.__module__
                if self.external_memory_bridge is not None
                else __name__
            ),
            "memory_enabled": self.context.memory_method != "no-memory",
            "gmemory_snapshot_id": snapshot_id,
            "result_schema_version": "e1-runtime-identity-v2",
        }
        if self.external_memory_bridge is not None:
            metadata["gmemory_upstream_components"] = self.external_memory_bridge.upstream_components
        if self.host_bridge is not None:
            metadata.update(self.host_bridge.metadata())
        return metadata

    def record_host_trace(
        self,
        *,
        sender: str,
        receiver: str,
        message_type: str,
        environment_action: str = "",
        observation: str = "",
        termination: bool = False,
    ) -> dict[str, Any]:
        if self.host_bridge is None:
            return {}
        event = self.host_bridge.trace_event(
            sender=sender,
            receiver=receiver,
            message_type=message_type,
            environment_action=environment_action,
            observation=observation,
            termination=termination,
        )
        self.host_trace.append(event)
        self._write_host_trace()
        return event

    def _write_host_trace(self) -> None:
        path = os.environ.get("TEAM_MEMORY_HOST_TRACE")
        if not path:
            return
        trace_path = Path(path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = trace_path.with_suffix(trace_path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "host_identity": self.host_bridge.metadata() if self.host_bridge else {},
                    "events": self.host_trace,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        tmp.replace(trace_path)

    def _append(self, entry: BlackboardEntry) -> dict[str, Any]:
        payload = self.adapter.on_agent_step(entry)
        self.blackboard_entries += 1
        divergence = payload["divergence"]
        requires_alignment = bool(divergence.get("requires_alignment"))
        if requires_alignment:
            signature = self._divergence_signature(entry.agent_id, divergence)
            if self.last_divergence_signature.get(entry.agent_id) == signature:
                self.duplicate_divergence_events += 1
            else:
                event_id = self._event_id(entry.agent_id, signature)
                self.divergence_events += 1
                self.active_divergence_events[entry.agent_id] = event_id
                self.last_divergence_signature[entry.agent_id] = signature
                for reason in divergence.get("reasons", []):
                    reason_key = str(reason)
                    self.divergence_reason_counts[reason_key] = (
                        self.divergence_reason_counts.get(reason_key, 0) + 1
                    )
                self.divergence_lifecycle.append(
                    {
                        "event_id": event_id,
                        "stage": "conflict_detected",
                        "agent_id": entry.agent_id,
                        "entry_id": payload.get("entry", {}).get("entry_id", entry.entry_id),
                        "reasons": list(divergence.get("reasons", [])),
                        "conflicting_state_keys": list(
                            divergence.get("conflicting_state_keys", [])
                        ),
                        "recommended_action": _text(
                            divergence.get("recommended_action", ""), 2_000
                        ),
                    }
                )
        return {
            **payload,
            "requires_alignment": requires_alignment,
            "active_divergence_event_id": self.active_divergence_events.get(entry.agent_id),
        }

    def _event_id(self, agent_id: str, signature: str) -> str:
        raw = f"{self.workspace.workspace_id}|{agent_id}|{self.blackboard_entries}|{signature}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _divergence_signature(agent_id: str, divergence: dict[str, Any]) -> str:
        payload = {
            "agent_id": agent_id,
            "reasons": sorted(str(item) for item in divergence.get("reasons", [])),
            "conflicting_state_keys": sorted(
                str(item) for item in divergence.get("conflicting_state_keys", [])
            ),
            "recommended_action": str(divergence.get("recommended_action", "")),
        }
        return hashlib.sha256(
            repr(sorted(payload.items())).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _prompt_prefix(sops: list[dict[str, Any]]) -> str:
        if not sops:
            return ""
        lines = ["Relevant shared SOP hint:"]
        for index, item in enumerate(sops[:1], 1):
            sop = item.get("sop", {})
            metadata = sop.get("metadata", {})
            procedure = sop.get("procedure", {})
            steps = procedure.get("steps", [])
            title = metadata.get("title", sop.get("sop_id", f"sop-{index}"))
            lines.append(f"{index}. {title}")
            for condition in metadata.get("context_conditions", []):
                lines.append(f"   Applies only when: {condition}")
            for warning in metadata.get("warnings", []):
                lines.append(f"   Warning: {warning}")
            for step in steps:
                instruction = step.get("instruction", "")
                if instruction:
                    lines.append(f"   - [{step.get('step_id', '')}] {instruction}")
                    if step.get("requires_confirmation"):
                        lines.append("     Requires user confirmation before execution.")
                    for condition in step.get("preconditions", []):
                        lines.append(f"     Precondition: {condition}")
            for edge in procedure.get("edges", []):
                lines.append(f"   Dependency: {edge['source']} -> {edge['target']}")
        return "\n".join(lines)
