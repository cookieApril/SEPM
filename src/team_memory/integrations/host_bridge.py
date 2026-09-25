"""Host identity and prompt-routing bridges for E1 host compositions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class HostRuntimeError(RuntimeError):
    """Raised when a requested host is not actually constructible."""


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


@dataclass(frozen=True)
class HostRuntimeBridge:
    """Concrete host identity used by official environment loops."""

    mas: str
    task: str
    project_root: Path
    roles: tuple[str, ...]
    routing_mode: str
    upstream_module: str
    upstream_commit: str

    @classmethod
    def build(cls, *, mas: str, task: str, project_root: Path) -> "HostRuntimeBridge":
        normalized = mas.lower()
        if normalized == "autogen":
            return cls(
                mas="autogen",
                task=task,
                project_root=project_root,
                roles=("solver", "ground_truth"),
                routing_mode="two-agent-solver-ground-truth",
                upstream_module="external/GMemory/tasks/mas_workflow/autogen/autogen.py::AutoGen",
                upstream_commit=_git_revision(project_root / "external/GMemory"),
            )
        if normalized == "dylan":
            dylan_path = project_root / "external/GMemory/tasks/mas_workflow/dylan/dylan.py"
            neuron_path = project_root / "external/GMemory/tasks/mas_workflow/dylan/neuron.py"
            if not dylan_path.is_file() or not neuron_path.is_file():
                raise HostRuntimeError(
                    "official DyLAN implementation is not present under external/GMemory/tasks/mas_workflow/dylan"
                )
            return cls(
                mas="dylan",
                task=task,
                project_root=project_root,
                roles=("solver", "critic", "planner"),
                routing_mode="dylan-neuron-grid-dynamic-routing",
                upstream_module="external/GMemory/tasks/mas_workflow/dylan/dylan.py::DyLAN",
                upstream_commit=_git_revision(project_root / "external/GMemory"),
            )
        raise HostRuntimeError(f"unsupported host MAS for E1 official smoke: {mas!r}")

    @property
    def adapter_identity(self) -> str:
        return (
            "team_memory.integrations.host_bridge.HostRuntimeBridge"
            f":{self.mas}:{self.upstream_module}:{self.upstream_commit}"
        )

    @property
    def topology_hash(self) -> str:
        graph = {
            "mas": self.mas,
            "task": self.task,
            "roles": self.roles,
            "routing_mode": self.routing_mode,
            "upstream_module": self.upstream_module,
            "upstream_commit": self.upstream_commit,
        }
        if self.mas == "dylan":
            graph["edges"] = [
                ("solver", "critic"),
                ("critic", "planner"),
                ("planner", "solver"),
                ("planner", "environment"),
            ]
        else:
            graph["edges"] = [("solver", "environment"), ("environment", "ground_truth")]
        return _hash_payload(graph)

    def actor_prefix(self, *, agent_id: str, observation: str = "") -> str:
        if self.mas != "dylan":
            return ""
        payload = {
            "selected_host": "DyLAN",
            "selected_role_graph": list(self.roles),
            "routing_mode": self.routing_mode,
            "agent_id": agent_id,
            "topology_hash": self.topology_hash,
        }
        if observation:
            payload["observation_hash"] = hashlib.sha256(
                observation.encode("utf-8", errors="ignore")
            ).hexdigest()[:16]
        return "DyLAN routing context:\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def metadata(self) -> dict[str, Any]:
        return {
            "host_execution_mode": f"{self.task}:{self.mas}:official-loop-host-bridge",
            "host_adapter_identity": self.adapter_identity,
            "host_adapter_class": self.__class__.__name__,
            "host_adapter_module": self.__class__.__module__,
            "host_topology_hash": self.topology_hash,
            "agent_roles": list(self.roles),
            "message_routing_mode": self.routing_mode,
            "host_upstream_module": self.upstream_module,
            "host_upstream_commit": self.upstream_commit,
        }

    def trace_event(
        self,
        *,
        sender: str,
        receiver: str,
        message_type: str,
        environment_action: str = "",
        observation: str = "",
        termination: bool = False,
    ) -> dict[str, Any]:
        return {
            "selected_role_node": receiver,
            "topology_revision": self.topology_hash,
            "sender": sender,
            "receiver": receiver,
            "routed_message_type": message_type,
            "environment_action": environment_action[:1000],
            "observation_hash": hashlib.sha256(observation.encode("utf-8", errors="ignore")).hexdigest()[:16],
            "termination_decision": bool(termination),
        }


def _git_revision(path: Path) -> str:
    head = path / ".git/HEAD"
    if not head.is_file():
        return "unknown"
    text = head.read_text(encoding="utf-8").strip()
    if text.startswith("ref: "):
        ref = path / ".git" / text.split(" ", 1)[1]
        return ref.read_text(encoding="utf-8").strip() if ref.is_file() else "unknown"
    return text
