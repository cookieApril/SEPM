"""Plug-in layer: text matching, failure lessons, frozen pools, textual control, vector store."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from sepm.alignment import ActionRecord, EmbeddingActionMatcher, derive_alignment
from sepm.config import MemoryConfig
from sepm.domains import IMPLICIT_MEMORY_FAILURE_MARKERS, text_domain
from sepm.embedding import HashingEmbedder
from sepm.episode import ProceduralEpisode, format_procedure_text, render_procedure
from sepm.models import CandidateStatus, PlanEdge, PlanNode, TaskPlan
from sepm.plugins._gmemory_compat import Document, LiteChroma
from sepm.service import SEPMService


class ScriptedSOPModel:
    """Returns canned JSON: a two-step candidate, then a failure lesson citing step 1."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        self.calls.append(user)
        if "FAILED" in user:
            return {"warnings": [
                {"text": "Answer briefly", "trigger_condition": "user asks a quick question", "evidence_steps": [1]},
                {"text": "Unsupported lesson", "trigger_condition": "", "evidence_steps": [2]},
            ]}
        return {
            "title": "Concise answers",
            "description": "Answer only what is asked.",
            "atomic_steps": [
                {"instruction": "answer the question directly", "evidence_steps": [2]},
                {"instruction": "omit unsolicited additions", "evidence_steps": [2]},
            ],
            "dependencies": [],
        }


def _service(tmp_path, **overrides: Any) -> SEPMService:
    config = replace(MemoryConfig(database_path=tmp_path / "pool.db"), **overrides)
    return SEPMService(config=config)


def _episode(service: SEPMService, model: ScriptedSOPModel, name: str = "w1") -> ProceduralEpisode:
    domain = text_domain("implicit_memory", HashingEmbedder(), failure_markers=IMPLICIT_MEMORY_FAILURE_MARKERS)
    return ProceduralEpisode(service, domain, workspace_id=name, task_goal="Assist the user", sop_model=model)


def test_unmatched_free_text_is_not_off_plan() -> None:
    matcher = EmbeddingActionMatcher(HashingEmbedder(), threshold=0.99)
    plan = TaskPlan(nodes=[PlanNode(node_id="n0", action="send the report")], edges=[])
    view = derive_alignment(plan, [ActionRecord("a long chatty reply about something else")], matcher)
    assert view.off_plan_actions == []


def test_prerequisite_violation_is_still_detected() -> None:
    matcher = EmbeddingActionMatcher(HashingEmbedder(), threshold=0.6)
    plan = TaskPlan(
        nodes=[PlanNode(node_id="a", action="clean the cup"), PlanNode(node_id="b", action="store the cup")],
        edges=[PlanEdge(source="a", target="b")],
    )
    view = derive_alignment(plan, [ActionRecord("store the cup")], matcher)
    assert view.missing_prerequisites == [("a", "b")]


def test_failed_steps_become_conditional_warnings(tmp_path) -> None:
    service, model = _service(tmp_path), ScriptedSOPModel()
    episode = _episode(service, model)
    episode.record_step("assistant", "39. Also, fun facts about fractions...", "[system] User feedback: just the answer")
    episode.record_step("assistant", "3.14", "[system] Success: concise answer")
    episode.finish(success=True, reward=1.0)
    candidates, _ = service.store.list_candidates(CandidateStatus.VALIDATED, limit=10, offset=0)
    assert len(candidates) == 1
    warnings = candidates[0].metadata.warnings
    # Only the lesson citing the failed step survives.
    assert [w.text for w in warnings] == ["Answer briefly"]
    assert warnings[0].evidence == ["w1#step-1"]


def test_frozen_pool_never_writes(tmp_path) -> None:
    service, model = _service(tmp_path, frozen_pool=True), ScriptedSOPModel()
    episode = _episode(service, model)
    episode.record_step("assistant", "3.14", "[system] Success")
    episode.finish(success=True, reward=1.0)
    for status in CandidateStatus:
        assert service.store.list_candidates(status, limit=10, offset=0)[1] == 0
    assert model.calls == []


def test_textual_control_has_no_graph_references(tmp_path) -> None:
    service, model = _service(tmp_path), ScriptedSOPModel()
    episode = _episode(service, model)
    episode.record_step("assistant", "39", "[system] Success")
    episode.record_step("assistant", "3.14", "[system] Success")
    episode.finish(success=True, reward=1.0)
    sop = service.retrieve_trial_candidates(
        "quick question", 1, context_facts={"task_family:implicit_memory"}, task_family="implicit_memory"
    )[0].sop
    text = format_procedure_text(sop)
    assert "answer the question directly" in text and "Prerequisite:" not in text and "1." not in text
    assert render_procedure(sop, True).startswith("1. ")


def test_textual_control_skips_task_graph(tmp_path) -> None:
    service, model = _service(tmp_path), ScriptedSOPModel()
    first = _episode(service, model, "w1")
    first.record_step("assistant", "39", "[system] Success")
    first.record_step("assistant", "3.14", "[system] Success")
    first.finish(success=True, reward=1.0)
    textual = _service(tmp_path, explicit_task_graph=False)
    second = _episode(textual, model, "w2")
    assert second.retrieve("quick question").trial is not None
    assert textual.store.get_workspace("w2").plan.nodes == []


def test_lite_chroma_matches_chroma_semantics(tmp_path) -> None:
    class Unit:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [self.embed_query(t) for t in texts]

        def embed_query(self, text: str) -> list[float]:
            return [1.0, 0.0] if "cup" in text else [0.0, 1.0]

    store = LiteChroma(embedding_function=Unit(), persist_directory=str(tmp_path))
    store.add_documents([Document("wash cup", {"label": True}), Document("open door", {"label": False})])
    (doc, distance), = store.similarity_search_with_score("cup", k=1)
    assert doc.page_content == "wash cup" and distance == pytest.approx(0.0)
    assert store.similarity_search("cup", k=5, filter={"label": False})[0].page_content == "open door"
    # Squared L2 between orthogonal unit vectors is 2, as in Chroma's default space.
    assert store.similarity_search_with_score("door", k=2)[1][1] == pytest.approx(2.0)
    reloaded = LiteChroma(embedding_function=Unit(), persist_directory=str(tmp_path))
    assert len(reloaded.get()["ids"]) == 2
