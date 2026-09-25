"""Local HotpotQA distractor preflight with resumable single-example execution."""

from __future__ import annotations

import json
import os
import re
import string
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_f1(prediction: str, answer: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(answer).split()
    if not predicted or not expected:
        return float(predicted == expected)
    common = sum(min(predicted.count(token), expected.count(token)) for token in set(predicted))
    if common == 0:
        return 0.0
    precision = common / len(predicted)
    recall = common / len(expected)
    return 2 * precision * recall / (precision + recall)


def load_hotpotqa(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"HotpotQA data must be a non-empty JSON list: {path}")
    required = {"_id", "question", "answer", "context", "supporting_facts"}
    for index, row in enumerate(payload):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"HotpotQA row {index} is missing fields: {sorted(missing)}")
    return payload


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", normalize_answer(text)))


@dataclass
class RetrievedPassage:
    title: str
    sentences: list[str]
    score: float

    @property
    def text(self) -> str:
        return "".join(self.sentences).strip()


class HotpotQARetriever:
    """Lexical retrieval over the ten distractor-setting candidate passages."""

    def __init__(self, context: list[list[Any]]) -> None:
        self.passages = [
            RetrievedPassage(str(title), [str(sentence) for sentence in sentences], 0.0)
            for title, sentences in context
        ]

    def search(self, query: str, *, top_k: int = 4) -> list[RetrievedPassage]:
        query_tokens = _tokens(query)
        ranked: list[RetrievedPassage] = []
        for passage in self.passages:
            title_tokens = _tokens(passage.title)
            body_tokens = _tokens(passage.text)
            score = 3.0 * len(query_tokens & title_tokens) + len(query_tokens & body_tokens)
            ranked.append(RetrievedPassage(passage.title, passage.sentences, score))
        return sorted(ranked, key=lambda item: (-item.score, item.title))[:top_k]

    @staticmethod
    def lookup(passages: list[RetrievedPassage], keyword: str) -> list[str]:
        needle = normalize_answer(keyword)
        return [
            sentence.strip()
            for passage in passages
            for sentence in passage.sentences
            if needle in normalize_answer(sentence)
        ]


class HotpotQAEnvironment:
    def __init__(self, example: dict[str, Any]) -> None:
        self.example = example
        self.retriever = HotpotQARetriever(example["context"])
        self.retrieved: list[RetrievedPassage] = []

    def search(self, query: str, *, top_k: int = 4) -> list[RetrievedPassage]:
        self.retrieved = self.retriever.search(query, top_k=top_k)
        return self.retrieved

    def lookup(self, keyword: str) -> list[str]:
        return self.retriever.lookup(self.retrieved, keyword)

    def finish(self, answer: str) -> dict[str, float]:
        gold = str(self.example["answer"])
        return {
            "exact_match": float(normalize_answer(answer) == normalize_answer(gold)),
            "f1": answer_f1(answer, gold),
        }


HOTPOTQA_SYSTEM_PROMPT = """Answer one HotpotQA question using only the supplied passages.
The answer may require combining facts from two passages. Return only the shortest exact
answer span; do not include reasoning, labels, or punctuation around the answer."""


def build_prompt(question: str, passages: list[RetrievedPassage]) -> str:
    evidence = "\n\n".join(f"[{item.title}] {item.text}" for item in passages)
    return f"Question: {question}\n\nCandidate passages:\n{evidence}\n\nAnswer:"


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@dataclass
class HotpotQARecorder:
    output: Path
    checkpoint: Path
    identity: dict[str, Any]
    cases: list[dict[str, Any]] = field(default_factory=list)

    def restore(self) -> bool:
        if not self.checkpoint.exists():
            return False
        payload = json.loads(self.checkpoint.read_text(encoding="utf-8"))
        if payload.get("identity") != self.identity:
            raise RuntimeError("HotpotQA checkpoint identity mismatch")
        self.cases = list(payload.get("cases", []))
        return bool(payload.get("complete"))

    def save(self, case: dict[str, Any], *, complete: bool = True) -> None:
        self.cases = [case]
        _atomic_write(
            self.checkpoint,
            {
                "version": 1,
                "identity": self.identity,
                "cases": self.cases,
                "complete": complete,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )

    def write_result(self, metadata: dict[str, Any]) -> None:
        case = self.cases[0]
        _atomic_write(
            self.output,
            {
                "context": self.identity,
                "metrics": {
                    "exact_match": case["exact_match"],
                    "f1": case["f1"],
                    "case_count": 1.0,
                    "primary_score": case["exact_match"],
                },
                "cases": self.cases,
                "metadata": metadata,
            },
        )


def run_local_preflight(
    *,
    data_path: Path,
    output: Path,
    checkpoint: Path,
    base_url: str,
    model: str,
    api_key: str = "EMPTY",
    case_index: int = 0,
) -> dict[str, Any]:
    examples = load_hotpotqa(data_path)
    example = examples[case_index]
    identity = {
        "benchmark": "hotpotqa-local-preflight",
        "dataset": "hotpot_dev_distractor_v1",
        "case_id": example["_id"],
        "case_index": case_index,
        "model": model,
        "base_url": base_url,
    }
    recorder = HotpotQARecorder(output, checkpoint, identity)
    resumed = recorder.restore()
    if not resumed:
        from openai import OpenAI

        environment = HotpotQAEnvironment(example)
        passages = environment.search(str(example["question"]), top_k=4)
        client = OpenAI(base_url=base_url, api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": HOTPOTQA_SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(str(example["question"]), passages)},
            ],
            temperature=0,
            max_tokens=64,
        )
        prediction = (response.choices[0].message.content or "").strip()
        scores = environment.finish(prediction)
        supporting_titles = {str(item[0]) for item in example["supporting_facts"]}
        retrieved_titles = [passage.title for passage in passages]
        recorder.save(
            {
                "case_id": example["_id"],
                "question": example["question"],
                "prediction": prediction,
                "answer": example["answer"],
                "retrieved_titles": retrieved_titles,
                "supporting_title_recall": len(supporting_titles & set(retrieved_titles))
                / len(supporting_titles),
                **scores,
            }
        )
    recorder.write_result(
        {
            "adapter": "team_memory.hotpotqa_local",
            "mode": "engineering-preflight-not-publication-evidence",
            "resumed_from_checkpoint": resumed,
            "search": "lexical ranking over distractor candidate passages",
            "answer_normalization": "lowercase, punctuation/articles removed, whitespace normalized",
        }
    )
    return json.loads(output.read_text(encoding="utf-8"))
