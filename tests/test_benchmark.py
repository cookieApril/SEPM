"""机制级 benchmark 的确定性回归测试。"""

from __future__ import annotations

import unittest

from team_memory.benchmark import (
    run_all_benchmarks,
    run_divergence_benchmark,
    run_evidence_benchmark,
    run_latency_benchmark,
    run_promotion_benchmark,
    run_retrieval_benchmark,
)


class BenchmarkTests(unittest.TestCase):
    def test_divergence_and_evidence_gold_cases(self) -> None:
        divergence = run_divergence_benchmark()["metrics"]
        self.assertEqual(divergence["accuracy"], 1.0)
        self.assertEqual(divergence["macro_signal_f1"], 1.0)
        self.assertEqual(divergence["false_alarm_rate"], 0.0)
        evidence = run_evidence_benchmark()["metrics"]
        self.assertEqual(evidence["accuracy"], 1.0)
        self.assertEqual(evidence["value_accuracy"], 1.0)
        self.assertEqual(evidence["abstention_accuracy"], 1.0)

    def test_promotion_rejects_unsafe_and_writes_first_valid_trial(self) -> None:
        metrics = run_promotion_benchmark()["metrics"]
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["unsafe_accept_rate"], 0.0)
        self.assertEqual(metrics["first_trial_write_success"], 1.0)

    def test_retrieval_reports_full_and_vector_baseline(self) -> None:
        report = run_retrieval_benchmark()
        self.assertIn("ndcg@3", report["full"])
        self.assertIn("ndcg@3", report["vector_only"])

    def test_latency_and_combined_report_are_serializable_shapes(self) -> None:
        self.assertEqual(run_latency_benchmark(iterations=2)["iterations"], 2.0)
        report = run_all_benchmarks(include_latency=False)
        self.assertEqual(report["benchmark"], "TeamMemoryBench")
        self.assertNotIn("latency", report["suites"])


if __name__ == "__main__":
    unittest.main()
