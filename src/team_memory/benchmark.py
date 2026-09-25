"""Team Memory 的机制级、可离线复现评测套件。

公开 Agent benchmark 通常同时混入模型能力、工具稳定性和环境随机性。本模块直接构造
带标准答案的最小场景，分别测偏差检测、证据裁决、SOP 晋升和 SOP 检索，使回归能够
定位到具体机制。性能套件只报告延迟/吞吐，不设置与机器绑定的通过阈值。
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import fmean, median
from time import perf_counter
from typing import Any

from .divergence import DivergenceDetector
from .embedding import HashingEmbedder, bounded_similarity
from .evidence import EvidenceResolver
from .models import (
    BlackboardEntry,
    BlackboardKind,
    CandidateStatus,
    Evidence,
    EvidenceTier,
    PlanNode,
    ProcedureGraph,
    ProcedureStep,
    ProposalOperation,
    ReproductionTrial,
    SOPCandidate,
    SOPMetadata,
    SOPVersion,
    TaskPlan,
    Workspace,
)
from .retrieval import SOPRetriever
from .service import TeamMemoryService


def _binary_metrics(expected: list[bool], predicted: list[bool]) -> dict[str, float]:
    """计算正类的 precision/recall/F1，并同时报告总体 accuracy。"""
    tp = sum(want and got for want, got in zip(expected, predicted))
    fp = sum(not want and got for want, got in zip(expected, predicted))
    fn = sum(want and not got for want, got in zip(expected, predicted))
    correct = sum(want == got for want, got in zip(expected, predicted))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "accuracy": correct / max(1, len(expected)),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
        "count": float(len(expected)),
    }


def _percentile(values: list[float], quantile: float) -> float:
    """使用 nearest-rank 变体计算小样本延迟分位数。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


@dataclass(frozen=True)
class DivergenceCase:
    case_id: str
    workspace: Workspace
    entry: BlackboardEntry
    conflicts: list[str]
    expected_reasons: frozenset[str]

    @property
    def should_align(self) -> bool:
        return bool(self.expected_reasons)


def _divergence_cases() -> list[DivergenceCase]:
    base_plan = TaskPlan(nodes=[PlanNode(node_id="search", action="search verified catalog")])

    def make(
        case_id: str,
        goal: str,
        plan: TaskPlan,
        conflicts: list[str],
        expected_reasons: set[str],
    ) -> DivergenceCase:
        workspace = Workspace(
            workspace_id=f"workspace-{case_id}",
            main_goal="find a safe restaurant",
            plan=base_plan,
        )
        entry = BlackboardEntry(
            workspace_id=workspace.workspace_id,
            agent_id="worker",
            kind=BlackboardKind.PLAN,
            task=goal,
            goal=goal,
            plan=plan,
        )
        return DivergenceCase(case_id, workspace, entry, conflicts, frozenset(expected_reasons))

    return [
        make("aligned", "find a safe restaurant", base_plan, [], set()),
        make(
            "goal-negation",
            "do not find a safe restaurant",
            base_plan,
            [],
            {"goal_divergence"},
        ),
        make(
            "plan-change",
            "find a safe restaurant",
            TaskPlan(nodes=[PlanNode(node_id="book", action="book immediately")]),
            [],
            {"plan_divergence"},
        ),
        make(
            "state-conflict",
            "find a safe restaurant",
            base_plan,
            ["booking.confirmed"],
            {"world_state_conflict"},
        ),
        make(
            "mixed-all",
            "do not find a safe restaurant",
            TaskPlan(nodes=[PlanNode(node_id="book", action="book immediately")]),
            ["booking.confirmed"],
            {"goal_divergence", "plan_divergence", "world_state_conflict"},
        ),
        make("aligned-repeat", "find a safe restaurant", base_plan, [], set()),
    ]


def run_divergence_benchmark() -> dict[str, Any]:
    """评测是否正确触发对齐，并保留逐 case 结果便于误差分析。"""
    detector = DivergenceDetector(HashingEmbedder())
    rows: list[dict[str, Any]] = []
    for case in _divergence_cases():
        report = detector.inspect(case.workspace, case.entry, case.conflicts)
        rows.append(
            {
                "case_id": case.case_id,
                "expected": case.should_align,
                "predicted": report.requires_alignment,
                "expected_reasons": sorted(case.expected_reasons),
                "reasons": report.reasons,
                "goal_divergence": report.goal_divergence,
                "plan_divergence": report.plan_divergence,
            }
        )
    metrics: dict[str, Any] = _binary_metrics(
        [row["expected"] for row in rows],
        [row["predicted"] for row in rows],
    )
    signal_metrics: dict[str, dict[str, float]] = {}
    for signal in ("goal_divergence", "plan_divergence", "world_state_conflict"):
        signal_metrics[signal] = _binary_metrics(
            [signal in row["expected_reasons"] for row in rows],
            [signal in row["reasons"] for row in rows],
        )
    clean_rows = [row for row in rows if not row["expected"]]
    metrics["false_alarm_rate"] = sum(row["predicted"] for row in clean_rows) / max(
        1, len(clean_rows)
    )
    metrics["macro_signal_f1"] = fmean(
        item["f1"] for item in signal_metrics.values()
    )
    metrics["signals"] = signal_metrics
    return {"metrics": metrics, "cases": rows}


@dataclass(frozen=True)
class EvidenceCase:
    case_id: str
    evidence: list[Evidence]
    expected_resolved: bool
    expected_value: Any = None


def _evidence_cases() -> list[EvidenceCase]:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [
        EvidenceCase(
            "authority-beats-inference",
            [
                Evidence(
                    tier=EvidenceTier.AGENT_INFERENCE,
                    source="agent",
                    value=False,
                    observed_at=now,
                ),
                Evidence(
                    tier=EvidenceTier.AUTHORITATIVE_STATE,
                    source="api",
                    value=True,
                    observed_at=now,
                ),
            ],
            True,
            True,
        ),
        EvidenceCase(
            "equal-tier-tie",
            [
                Evidence(
                    tier=EvidenceTier.AUTHORITATIVE_STATE,
                    source="api-a",
                    value=True,
                    observed_at=now,
                ),
                Evidence(
                    tier=EvidenceTier.AUTHORITATIVE_STATE,
                    source="api-b",
                    value=False,
                    observed_at=now,
                ),
            ],
            False,
        ),
        EvidenceCase(
            "newer-observation-wins",
            [
                Evidence(
                    tier=EvidenceTier.TOOL_OBSERVATION,
                    source="same-state-tool",
                    value=False,
                    observed_at=now,
                ),
                Evidence(
                    tier=EvidenceTier.TOOL_OBSERVATION,
                    source="same-state-tool",
                    value=True,
                    observed_at=now + timedelta(seconds=1),
                ),
            ],
            True,
            True,
        ),
        EvidenceCase(
            "low-confidence-abstains",
            [
                Evidence(
                    tier=EvidenceTier.AUTHORITATIVE_STATE,
                    source="noisy-api",
                    value=True,
                    observed_at=now,
                    confidence=0.2,
                )
            ],
            False,
        ),
        EvidenceCase("missing-evidence-abstains", [], False),
    ]


def run_evidence_benchmark() -> dict[str, Any]:
    """评测 resolved/abstain 决策及已解决 case 的值准确率。"""
    resolver = EvidenceResolver()
    rows: list[dict[str, Any]] = []
    for case in _evidence_cases():
        result = resolver.resolve_world_state("benchmark.state", case.evidence)
        rows.append(
            {
                "case_id": case.case_id,
                "expected_resolved": case.expected_resolved,
                "predicted_resolved": result.resolved,
                "expected_value": case.expected_value,
                "predicted_value": result.value,
                "next_action": result.next_action,
            }
        )
    resolution = _binary_metrics(
        [row["expected_resolved"] for row in rows],
        [row["predicted_resolved"] for row in rows],
    )
    resolved_rows = [row for row in rows if row["expected_resolved"]]
    value_accuracy = sum(
        row["predicted_resolved"] and row["predicted_value"] == row["expected_value"]
        for row in resolved_rows
    ) / max(1, len(resolved_rows))
    abstention_rows = [row for row in rows if not row["expected_resolved"]]
    abstention_accuracy = sum(not row["predicted_resolved"] for row in abstention_rows) / max(
        1, len(abstention_rows)
    )
    return {
        "metrics": {
            **resolution,
            "value_accuracy": value_accuracy,
            "abstention_accuracy": abstention_accuracy,
        },
        "cases": rows,
    }


@dataclass(frozen=True)
class PromotionCase:
    case_id: str
    safe: bool = True
    candidate_state_verified: bool = True
    causal_confidence: float = 0.9
    trial_success: bool = True
    trial_state_verified: bool = True
    causal_supported: bool = True
    candidate_reward: float = 1.0
    baseline_reward: float = 0.2
    cost: float = 0.1
    should_promote: bool = True


def _promotion_cases() -> list[PromotionCase]:
    return [
        PromotionCase("valid-first-trial"),
        PromotionCase("unsafe-payment", safe=False, should_promote=False),
        PromotionCase(
            "candidate-state-unverified",
            candidate_state_verified=False,
            should_promote=False,
        ),
        PromotionCase("low-causal-confidence", causal_confidence=0.2, should_promote=False),
        PromotionCase("trial-failed", trial_success=False, should_promote=False),
        PromotionCase("trial-state-unverified", trial_state_verified=False, should_promote=False),
        PromotionCase("causal-unsupported", causal_supported=False, should_promote=False),
        PromotionCase(
            "negative-net-benefit",
            candidate_reward=0.2,
            baseline_reward=0.3,
            cost=0.1,
            should_promote=False,
        ),
    ]


def _run_promotion_case(case: PromotionCase) -> dict[str, Any]:
    """通过真实 TeamMemoryService 和临时 SQLite 执行一个完整晋升 case。"""
    with tempfile.TemporaryDirectory() as directory:
        service = TeamMemoryService(Path(directory) / "benchmark.db")
        workspace = Workspace(
            workspace_id=f"workspace-{case.case_id}",
            main_goal="validate procedure",
        )
        service.create_workspace(workspace)
        step = ProcedureStep(
            instruction="Pay immediately" if not case.safe else "Verify the artifact before use",
            action_type="payment" if not case.safe else "verify",
        )
        candidate = SOPCandidate(
            operation=ProposalOperation.CREATE,
            procedure=ProcedureGraph(steps=[step]),
            metadata=SOPMetadata(title=case.case_id, task_family="benchmark"),
            source_workspace_ids=[workspace.workspace_id],
            state_verified=case.candidate_state_verified,
            causal_confidence=case.causal_confidence,
        )
        stored, safety = service.propose_sop(candidate)
        started = perf_counter()
        service.record_reproduction(
            ReproductionTrial(
                candidate_id=stored.candidate_id,
                workspace_id=workspace.workspace_id,
                task_family="benchmark",
                environment_fingerprint="deterministic-v1",
                success=case.trial_success,
                baseline_reward=case.baseline_reward,
                candidate_reward=case.candidate_reward,
                cost=case.cost,
                state_verified=case.trial_state_verified,
                causal_supported=case.causal_supported,
            )
        )
        elapsed_ms = (perf_counter() - started) * 1_000
        final = service.store.get_candidate(stored.candidate_id)
        promoted = final.status == CandidateStatus.PROMOTED
        return {
            "case_id": case.case_id,
            "expected": case.should_promote,
            "predicted": promoted,
            "safe": safety.allowed,
            "candidate_status": final.status.value,
            "elapsed_ms": elapsed_ms,
        }


def run_promotion_benchmark() -> dict[str, Any]:
    """评测完整业务路径，而不是重新实现一份门控规则。"""
    rows = [_run_promotion_case(case) for case in _promotion_cases()]
    metrics = _binary_metrics(
        [row["expected"] for row in rows],
        [row["predicted"] for row in rows],
    )
    unsafe = [row for row in rows if not row["safe"]]
    valid_first_trial = next(row for row in rows if row["case_id"] == "valid-first-trial")
    metrics.update(
        {
            "unsafe_accept_rate": sum(row["predicted"] for row in unsafe) / max(1, len(unsafe)),
            "first_trial_write_success": float(valid_first_trial["predicted"]),
            "mean_case_latency_ms": fmean(row["elapsed_ms"] for row in rows),
        }
    )
    return {"metrics": metrics, "cases": rows}


def _sop(
    sop_id: str,
    title: str,
    action: str,
    tags: list[str],
    *,
    safety_class: str = "normal",
) -> SOPVersion:
    step = ProcedureStep(step_id=f"{sop_id}-step", instruction=title, action_type=action)
    return SOPVersion(
        sop_id=sop_id,
        version=1,
        procedure=ProcedureGraph(steps=[step]),
        metadata=SOPMetadata(
            title=title,
            description=f"Standard procedure for {action}",
            task_family=action,
            applicability=tags,
            tags=tags,
            safety_class=safety_class,
        ),
        success_count=8,
        failure_count=2,
        validation_score=0.9,
        retrieval_count=5,
    )


def _retrieval_corpus() -> list[SOPVersion]:
    return [
        _sop("catalog", "Search and rank a verified catalog", "search", ["catalog", "ranking"]),
        _sop(
            "payment",
            "Confirm authorization before payment",
            "payment",
            ["payment", "authorization", "confirmation"],
            safety_class="critical",
        ),
        _sop(
            "migration",
            "Create a snapshot before database migration",
            "migration",
            ["database", "backup", "snapshot"],
            safety_class="sensitive",
        ),
        _sop("debug", "Reproduce and isolate a code failure", "debug", ["code", "test", "failure"]),
    ]


@dataclass(frozen=True)
class RetrievalCase:
    case_id: str
    query: str
    relevant_sop_ids: set[str]
    query_plan: TaskPlan


def _retrieval_cases() -> list[RetrievalCase]:
    return [
        RetrievalCase(
            "catalog-query",
            "rank verified catalog results",
            {"catalog"},
            TaskPlan(nodes=[PlanNode(node_id="catalog-step", action="search")]),
        ),
        RetrievalCase(
            "payment-query",
            "confirm authorization before payment",
            {"payment"},
            TaskPlan(nodes=[PlanNode(node_id="payment-step", action="payment")]),
        ),
        RetrievalCase(
            "migration-query",
            "backup snapshot for database migration",
            {"migration"},
            TaskPlan(nodes=[PlanNode(node_id="migration-step", action="migration")]),
        ),
        RetrievalCase(
            "debug-query",
            "debug code test failure",
            {"debug"},
            TaskPlan(nodes=[PlanNode(node_id="debug-step", action="debug")]),
        ),
        # 文本刻意不提供区分信息，用来测图结构是否能胜过 vector-only tie breaking。
        RetrievalCase(
            "ambiguous-graph-query",
            "standard procedure",
            {"payment"},
            TaskPlan(nodes=[PlanNode(node_id="payment-step", action="payment")]),
        ),
    ]


def _vector_only_rank(query: str, corpus: list[SOPVersion]) -> list[str]:
    embedder = HashingEmbedder()

    def document(sop: SOPVersion) -> str:
        return " ".join(
            [
                sop.metadata.title,
                sop.metadata.description,
                sop.metadata.task_family,
                *sop.metadata.applicability,
                *sop.metadata.tags,
                *(step.instruction for step in sop.procedure.steps),
            ]
        )

    return [
        sop.sop_id
        for sop in sorted(
            corpus,
            key=lambda item: bounded_similarity(query, document(item), embedder),
            reverse=True,
        )
    ]


def _ranking_metrics(
    rankings: list[list[str]],
    relevant: list[set[str]],
    k: int = 3,
) -> dict[str, float]:
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    for ranking, expected in zip(rankings, relevant):
        top_k = ranking[:k]
        recalls.append(len(expected & set(top_k)) / max(1, len(expected)))
        first_rank = next(
            (index + 1 for index, item in enumerate(ranking) if item in expected),
            None,
        )
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
        dcg = sum(
            (1.0 if item in expected else 0.0) / math.log2(index + 2)
            for index, item in enumerate(top_k)
        )
        ideal_hits = min(k, len(expected))
        idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
        ndcgs.append(dcg / max(1e-12, idcg))
    return {
        f"recall@{k}": fmean(recalls),
        "mrr": fmean(reciprocal_ranks),
        f"ndcg@{k}": fmean(ndcgs),
        "count": float(len(rankings)),
    }


def run_retrieval_benchmark() -> dict[str, Any]:
    """比较 BM25+VectorSearch 混合检索和 vector-only 基线。"""
    corpus = _retrieval_corpus()
    cases = _retrieval_cases()
    retriever = SOPRetriever(HashingEmbedder())
    full_rankings = [
        [
            item.sop.sop_id
            for item in retriever.rank(case.query, corpus, len(corpus), case.query_plan)
        ]
        for case in cases
    ]
    vector_rankings = [_vector_only_rank(case.query, corpus) for case in cases]
    relevant = [case.relevant_sop_ids for case in cases]
    rows = [
        {
            "case_id": case.case_id,
            "relevant": sorted(case.relevant_sop_ids),
            "full_ranking": full_rankings[index],
            "vector_only_ranking": vector_rankings[index],
        }
        for index, case in enumerate(cases)
    ]
    return {
        "full": _ranking_metrics(full_rankings, relevant),
        "vector_only": _ranking_metrics(vector_rankings, relevant),
        "cases": rows,
    }


def run_latency_benchmark(iterations: int = 30) -> dict[str, float]:
    """测量首个成功 trial 从写入到 SOP 可读的延迟；不设跨机器硬阈值。"""
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    durations: list[float] = []
    with tempfile.TemporaryDirectory() as directory:
        service = TeamMemoryService(Path(directory) / "latency.db")
        for index in range(iterations):
            workspace = Workspace(workspace_id=f"latency-{index}", main_goal="benchmark latency")
            service.create_workspace(workspace)
            candidate, _ = service.propose_sop(
                SOPCandidate(
                    operation=ProposalOperation.CREATE,
                    procedure=ProcedureGraph(steps=[ProcedureStep(instruction="Verify result")]),
                    metadata=SOPMetadata(title=f"latency-{index}", task_family="latency"),
                    source_workspace_ids=[workspace.workspace_id],
                    state_verified=True,
                    causal_confidence=0.9,
                )
            )
            started = perf_counter()
            service.record_reproduction(
                ReproductionTrial(
                    candidate_id=candidate.candidate_id,
                    workspace_id=workspace.workspace_id,
                    task_family="latency",
                    environment_fingerprint="deterministic-v1",
                    success=True,
                    baseline_reward=0.0,
                    candidate_reward=1.0,
                    cost=0.0,
                    state_verified=True,
                    causal_supported=True,
                )
            )
            service.store.get_sop(candidate.candidate_id)
            durations.append((perf_counter() - started) * 1_000)
    total_seconds = sum(durations) / 1_000
    return {
        "iterations": float(iterations),
        "mean_ms": fmean(durations),
        "median_ms": median(durations),
        "p50_ms": _percentile(durations, 0.50),
        "p95_ms": _percentile(durations, 0.95),
        "writes_per_second": iterations / max(1e-12, total_seconds),
    }


def run_all_benchmarks(*, include_latency: bool = True, iterations: int = 30) -> dict[str, Any]:
    """运行完整 suite，返回可直接 JSON 序列化的报告。"""
    suites: dict[str, Any] = {
        "divergence": run_divergence_benchmark(),
        "evidence": run_evidence_benchmark(),
        "promotion": run_promotion_benchmark(),
        "retrieval": run_retrieval_benchmark(),
    }
    if include_latency:
        suites["latency"] = {"metrics": run_latency_benchmark(iterations)}
    return {
        "benchmark": "TeamMemoryBench",
        "schema_version": 1,
        "suites": suites,
    }


def main() -> None:
    """命令行入口：运行 suite，并将完整 case 结果输出到 stdout 或 JSON 文件。"""
    parser = argparse.ArgumentParser(description="Run deterministic Team Memory benchmarks")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--skip-latency", action="store_true", help="Skip machine-dependent timing")
    parser.add_argument("--iterations", type=int, default=30, help="Latency benchmark iterations")
    args = parser.parse_args()
    report = run_all_benchmarks(include_latency=not args.skip_latency, iterations=args.iterations)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
