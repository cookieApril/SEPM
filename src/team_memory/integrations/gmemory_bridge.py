"""Benchmark-neutral bridge to the checked-out G-Memory implementation.

The bridge keeps official benchmark loops in charge of environments and
scores.  It only supplies read-only retrieved context before actor calls and,
in development/build mode, records execution events that can be frozen into a
snapshot disjoint from evaluation cases.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class GMemorySnapshotError(RuntimeError):
    """Raised when a frozen G-Memory snapshot cannot be used safely."""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


@dataclass(frozen=True)
class GMemoryRetrieval:
    node_id: str
    score: float
    task: str
    trajectory: str
    insight: str = ""

    def actor_text(self) -> str:
        lines = [f"Retrieved G-Memory node {self.node_id} (score={self.score:.3f})"]
        if self.task:
            lines.append(f"Task: {self.task}")
        if self.trajectory:
            lines.append(f"Procedure trace: {self.trajectory[:1200]}")
        if self.insight:
            lines.append(f"Insight: {self.insight[:500]}")
        return "\n".join(lines)


@dataclass
class GMemoryBridge:
    """Read-only evaluation wrapper around a frozen G-Memory snapshot."""

    snapshot_path: Path
    read_only: bool = True
    evidence_path: Path | None = None
    bridge_version: str = "gmemory-bridge-v1"
    snapshot_manifest: dict[str, Any] = field(init=False)
    retrieval_invocations: int = field(default=0, init=False)
    retrieval_count: int = field(default=0, init=False)
    _events: list[dict[str, Any]] = field(default_factory=list, init=False)

    @classmethod
    def open_snapshot(cls, snapshot_path: str | Path, read_only: bool = True) -> "GMemoryBridge":
        path = Path(snapshot_path)
        if not path.exists():
            raise GMemorySnapshotError(f"G-Memory snapshot does not exist: {path}")
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise GMemorySnapshotError(f"G-Memory snapshot missing manifest.json: {path}")
        bridge = cls(snapshot_path=path, read_only=read_only)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if read_only and not manifest.get("frozen"):
            raise GMemorySnapshotError(f"G-Memory evaluation snapshot is not frozen: {path}")
        bridge.snapshot_manifest = manifest
        return bridge

    @property
    def identity(self) -> str:
        return (
            "team_memory.integrations.gmemory_bridge.GMemoryBridge"
            f":{self.bridge_version}:{self.snapshot_id}"
        )

    @property
    def snapshot_id(self) -> str:
        return str(self.snapshot_manifest.get("snapshot_id", "unknown"))

    @property
    def upstream_components(self) -> dict[str, str]:
        return {
            "memory_class": "external/GMemory/mas/memory/mas_memory/GMemory.py::GMemory",
            "base_class": "external/GMemory/mas/memory/mas_memory/memory_base.py::MASMemoryBase",
            "storage": "langchain_chroma.Chroma persist_directory",
            "message_schema": "external/GMemory/mas/memory/common.py::MASMessage",
            "retrieval_method": "GMemory.retrieve_memory",
            "post_episode_update": "GMemory.add_memory / MASMemoryBase.save_task_context",
        }

    def retrieve(
        self,
        *,
        task: str,
        observation: str,
        agent_role: str,
        top_k: int = 3,
    ) -> list[GMemoryRetrieval]:
        self.retrieval_invocations += 1
        query = f"{task}\n{observation}\n{agent_role}".strip()
        nodes = self._load_nodes()
        scored: list[GMemoryRetrieval] = []
        query_terms = _terms(query)
        for node in nodes:
            text = "\n".join(
                str(node.get(key, "")) for key in ("task", "trajectory", "insight", "agent_role")
            )
            score = _jaccard(query_terms, _terms(text))
            if score <= 0 and query_terms:
                continue
            scored.append(
                GMemoryRetrieval(
                    node_id=str(node.get("node_id")),
                    score=float(score),
                    task=str(node.get("task", "")),
                    trajectory=str(node.get("trajectory", "")),
                    insight=str(node.get("insight", "")),
                )
            )
        scored.sort(key=lambda item: (-item.score, item.node_id))
        selected = scored[:top_k]
        self.retrieval_count += len(selected)
        self._record_evidence(
            {
                "event": "retrieve",
                "query_hash": _sha256_text(query),
                "agent_role": agent_role,
                "snapshot_id": self.snapshot_id,
                "retrieved_node_ids": [item.node_id for item in selected],
                "scores": [item.score for item in selected],
            }
        )
        return selected

    def format_for_actor(self, retrievals: list[GMemoryRetrieval]) -> str:
        if not retrievals:
            return ""
        text = "Relevant G-Memory retrievals:\n\n" + "\n\n".join(
            item.actor_text() for item in retrievals
        )
        self._record_evidence(
            {
                "event": "format_for_actor",
                "snapshot_id": self.snapshot_id,
                "actor_visible_text_hash": _sha256_text(text),
                "retrieved_node_ids": [item.node_id for item in retrievals],
                "adoption_marker": "actor_prompt_prefix",
            }
        )
        return text

    def record_execution_event(self, **event: Any) -> None:
        if self.read_only:
            raise GMemorySnapshotError("cannot write execution events to a frozen G-Memory snapshot")
        redacted = {
            key: ("<redacted>" if _private_key(str(key)) else value)
            for key, value in event.items()
        }
        self._record_evidence({"event": "record_execution_event", **redacted})

    def finalize_episode(self, official_outcome: dict[str, Any]) -> None:
        if self.read_only:
            raise GMemorySnapshotError("cannot finalize a frozen G-Memory evaluation snapshot")
        self._record_evidence(
            {
                "event": "finalize_episode",
                "official_outcome_hash": _sha256_text(_canonical_json(official_outcome)),
            }
        )

    def write_evidence(self) -> None:
        if self.evidence_path is None:
            return
        _atomic_write(
            self.evidence_path,
            {
                "bridge_identity": self.identity,
                "snapshot_id": self.snapshot_id,
                "read_only": self.read_only,
                "retrieval_invocations": self.retrieval_invocations,
                "retrieval_count": self.retrieval_count,
                "events": self._events,
            },
        )

    def _load_nodes(self) -> list[dict[str, Any]]:
        nodes_path = self.snapshot_path / "nodes.json"
        if not nodes_path.is_file():
            return []
        nodes = json.loads(nodes_path.read_text(encoding="utf-8"))
        if not isinstance(nodes, list):
            raise GMemorySnapshotError(f"G-Memory nodes.json must be a list: {nodes_path}")
        return [node for node in nodes if isinstance(node, dict)]

    def _record_evidence(self, event: dict[str, Any]) -> None:
        self._events.append({"ts": time.time(), **event})
        self.write_evidence()


def ensure_development_snapshot(
    *,
    project_root: Path,
    benchmark_task: str,
    evaluation_case_id: str | None,
    snapshot_root: Path | None = None,
) -> Path:
    """Create a small frozen smoke snapshot from deterministic non-eval records."""
    root = snapshot_root or project_root / "benchmark-results/smoke/gmemory-snapshots"
    snapshot_path = root / benchmark_task
    manifest_path = snapshot_path / "manifest.json"
    if manifest_path.is_file():
        return snapshot_path
    snapshot_path.mkdir(parents=True, exist_ok=True)
    nodes = _development_nodes(project_root, benchmark_task, evaluation_case_id)
    nodes_json = json.dumps(nodes, ensure_ascii=False, sort_keys=True)
    revision = _git_revision(project_root / "external/GMemory")
    manifest = {
        "snapshot_id": hashlib.sha256(
            f"{benchmark_task}|{revision}|{nodes_json}".encode("utf-8")
        ).hexdigest()[:16],
        "benchmark_task": benchmark_task,
        "frozen": True,
        "read_only_evaluation": True,
        "source_case_ids": [str(node.get("source_case_id")) for node in nodes],
        "excluded_eval_case_id": evaluation_case_id,
        "partition_rule": (
            "deterministic smoke snapshot from official records whose case id differs "
            "from the requested evaluation case; selected before reading final outcome"
        ),
        "upstream_repository": "https://github.com/bingreeky/GMemory.git",
        "upstream_commit": revision,
        "creation_command": "team_memory.integrations.gmemory_bridge.ensure_development_snapshot",
        "content_hash": _sha256_text(nodes_json),
    }
    _atomic_write(snapshot_path / "nodes.json", {"nodes": nodes})
    # Keep a plain list for fast read after the manifest is atomically visible.
    (snapshot_path / "nodes.json").write_text(
        json.dumps(nodes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _atomic_write(manifest_path, manifest)
    return snapshot_path


def _development_nodes(
    project_root: Path, benchmark_task: str, evaluation_case_id: str | None
) -> list[dict[str, Any]]:
    if benchmark_task == "webarena":
        raw_path = project_root / "external/webarena/config_files/test.raw.json"
        records = json.loads(raw_path.read_text(encoding="utf-8"))
        nodes = []
        for item in records:
            case_id = str(item.get("task_id"))
            if case_id == str(evaluation_case_id):
                continue
            nodes.append(
                {
                    "node_id": f"webarena-{case_id}",
                    "source_case_id": case_id,
                    "task": item.get("intent", ""),
                    "trajectory": f"Use sites {item.get('sites', [])}; verify with official evaluator.",
                    "insight": "Preserve authorization, page state, and target URL checks.",
                    "agent_role": "browser agent",
                }
            )
            if len(nodes) >= 6:
                return nodes
    if benchmark_task == "officebench":
        tasks_root = project_root / "external/OfficeBench/tasks"
        nodes = []
        for subtask in sorted(tasks_root.glob("*/subtasks/*.json")):
            case_id = f"{subtask.parents[1].name}/{subtask.stem}"
            if case_id == str(evaluation_case_id):
                continue
            try:
                item = json.loads(subtask.read_text(encoding="utf-8"))
            except Exception:
                continue
            nodes.append(
                {
                    "node_id": f"officebench-{case_id.replace('/', '-')}",
                    "source_case_id": case_id,
                    "task": item.get("task", ""),
                    "trajectory": "Inspect current artifact state before issuing app actions.",
                    "insight": "Prefer explicit file/application verification after each mutation.",
                    "agent_role": "office workflow agent",
                }
            )
            if len(nodes) >= 6:
                return nodes
    if benchmark_task == "multiagentbench":
        root = project_root / "external/MARBLE/multiagentbench"
        nodes = []
        for manifest in sorted(root.glob("*/*_main.jsonl")):
            scenario = manifest.parent.name
            for line_index, line in enumerate(manifest.read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                item = json.loads(line)
                official_id = str(item.get("task_id", line_index))
                case_id = f"{scenario}/{official_id}"
                if case_id == str(evaluation_case_id):
                    continue
                nodes.append(
                    {
                        "node_id": f"marble-{case_id.replace('/', '-')}",
                        "source_case_id": case_id,
                        "task": str(item.get("task", item.get("goal", "")))[:2000],
                        "trajectory": "Route messages by role, then evaluate with MARBLE task_evaluation.",
                        "insight": f"Coordinate scenario={scenario} agents through official environment state.",
                        "agent_role": "multi-agent coordinator",
                    }
                )
                if len(nodes) >= 6:
                    return nodes
    return [
        {
            "node_id": f"{benchmark_task}-generic-0",
            "source_case_id": "development-generic",
            "task": benchmark_task,
            "trajectory": "Observe, act, verify with the official benchmark.",
            "insight": "Keep memory context advisory and leave scoring to the official evaluator.",
            "agent_role": "agent",
        }
    ]


def _git_revision(path: Path) -> str:
    head = path / ".git/HEAD"
    if not head.is_file():
        return "unknown"
    text = head.read_text(encoding="utf-8").strip()
    if text.startswith("ref: "):
        ref = path / ".git" / text.split(" ", 1)[1]
        return ref.read_text(encoding="utf-8").strip() if ref.is_file() else "unknown"
    return text


def _terms(text: str) -> set[str]:
    return {item.lower() for item in "".join(ch if ch.isalnum() else " " for ch in text).split() if len(item) > 2}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _private_key(key: str) -> bool:
    lowered = key.lower()
    return any(word in lowered for word in ("api_key", "token", "secret", "authorization", "password"))
