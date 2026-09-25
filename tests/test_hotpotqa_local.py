import json
import tempfile
import unittest
from pathlib import Path

from team_memory.hotpotqa_local import (
    HotpotQAEnvironment,
    HotpotQARecorder,
    answer_f1,
    normalize_answer,
)


EXAMPLE = {
    "_id": "example-1",
    "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
    "answer": "yes",
    "context": [
        ["Scott Derrickson", ["Scott Derrickson is an American filmmaker."]],
        ["Ed Wood", ["Ed Wood was an American filmmaker."]],
        ["Distractor", ["This passage is unrelated."]],
    ],
    "supporting_facts": [["Scott Derrickson", 0], ["Ed Wood", 0]],
}


class HotpotQALocalTests(unittest.TestCase):
    def test_retrieval_lookup_and_normalized_scoring(self) -> None:
        environment = HotpotQAEnvironment(EXAMPLE)
        passages = environment.search(EXAMPLE["question"], top_k=2)
        self.assertEqual({item.title for item in passages}, {"Scott Derrickson", "Ed Wood"})
        self.assertIn("American filmmaker", environment.lookup("American")[0])
        self.assertEqual(environment.finish("Yes."), {"exact_match": 1.0, "f1": 1.0})
        self.assertEqual(normalize_answer("The United States."), "united states")
        self.assertEqual(answer_f1("United States", "the United States"), 1.0)

    def test_checkpoint_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = {"case_id": "example-1", "model": "local"}
            recorder = HotpotQARecorder(root / "result.json", root / "checkpoint.json", identity)
            recorder.save({"exact_match": 1.0, "f1": 1.0})
            restored = HotpotQARecorder(root / "result.json", root / "checkpoint.json", identity)
            self.assertTrue(restored.restore())
            self.assertEqual(restored.cases[0]["exact_match"], 1.0)
            mismatched = HotpotQARecorder(
                root / "result.json", root / "checkpoint.json", {"case_id": "other"}
            )
            with self.assertRaises(RuntimeError):
                mismatched.restore()


if __name__ == "__main__":
    unittest.main()
