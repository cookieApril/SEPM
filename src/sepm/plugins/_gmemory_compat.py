"""Import G-Memory's official memory classes without its heavyweight runtime.

G-Memory's ``mas`` package imports ``sentence_transformers`` (torch) and
``langchain_chroma`` at module level and prints the API key when ``mas.llm`` is
imported. On a small evaluation machine we keep the official memory algorithms
unchanged and replace only infrastructure:

* ``LiteChroma`` implements the subset of the Chroma vector-store API that the
  memories call (``add_documents``, ``similarity_search[_with_score]`` with an
  equality metadata filter, ``get``) with Chroma's default squared-L2 distance,
  so thresholds such as ``similarity = 1 - distance`` behave as upstream;
* embeddings come from the caller (an API embedder) instead of a local model;
* import-time stdout from ``mas.llm`` is discarded so secrets never reach logs.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import threading
import types
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_IMPORT_LOCK = threading.Lock()
_LOADED: dict[str, Any] = {}


@dataclass
class Document:
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str | None = None

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Document)
            and self.page_content == other.page_content
            and self.metadata == other.metadata
        )


class LiteChroma:
    """In-memory stand-in for ``langchain_chroma.Chroma`` (squared-L2 distance)."""

    def __init__(self, embedding_function: Any = None, persist_directory: str | None = None, **_: Any) -> None:
        self._embedding_function = embedding_function
        self._ids: list[str] = []
        self._docs: list[Document] = []
        self._vectors: list[list[float]] = []
        self._lock = threading.Lock()
        self._path = Path(persist_directory) / "lite_chroma.json" if persist_directory else None
        if self._path is not None and self._path.exists():
            # Resume a persisted pool, as Chroma does from its persist_directory.
            for row in json.loads(self._path.read_text(encoding="utf-8")):
                self._ids.append(row["id"])
                self._docs.append(Document(row["page_content"], row["metadata"], row["id"]))
                self._vectors.append(row["vector"])

    def add_documents(self, documents: list[Document], **_: Any) -> list[str]:
        vectors = self._embedding_function.embed_documents([doc.page_content for doc in documents])
        ids = [str(uuid.uuid4()) for _ in documents]
        with self._lock:
            for doc_id, doc, vector in zip(ids, documents, vectors):
                self._ids.append(doc_id)
                self._docs.append(Document(doc.page_content, dict(doc.metadata), doc_id))
                self._vectors.append(list(vector))
            self._persist()
        return ids

    def _persist(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"id": doc_id, "page_content": doc.page_content, "metadata": doc.metadata, "vector": vector}
            for doc_id, doc, vector in zip(self._ids, self._docs, self._vectors)
        ]
        temporary = self._path.with_suffix(".tmp")
        temporary.write_text(json.dumps(rows), encoding="utf-8")
        temporary.replace(self._path)

    @staticmethod
    def _matches(metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
        return not where or all(metadata.get(key) == value for key, value in where.items())

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: dict[str, Any] | None = None, **_: Any  # noqa: A002
    ) -> list[tuple[Document, float]]:
        if k <= 0:
            return []
        with self._lock:
            pool = [
                (doc, vector) for doc, vector in zip(self._docs, self._vectors)
                if self._matches(doc.metadata, filter)
            ]
        if not pool:
            return []
        query_vector = self._embedding_function.embed_query(query)
        scored = [
            (doc, sum((a - b) ** 2 for a, b in zip(query_vector, vector)))
            for doc, vector in pool
        ]
        scored.sort(key=lambda item: item[1])
        return scored[:k]

    def similarity_search(
        self, query: str, k: int = 4, filter: dict[str, Any] | None = None, **kwargs: Any  # noqa: A002
    ) -> list[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k, filter, **kwargs)]

    def get(self, ids: list[str] | None = None, **_: Any) -> dict[str, list[Any]]:
        with self._lock:
            rows = [
                (doc_id, doc) for doc_id, doc in zip(self._ids, self._docs)
                if ids is None or doc_id in ids
            ]
        return {
            "ids": [doc_id for doc_id, _ in rows],
            "documents": [doc.page_content for _, doc in rows],
            "metadatas": [dict(doc.metadata) for _, doc in rows],
        }


def _install_shims() -> None:
    chroma = types.ModuleType("langchain_chroma")
    chroma.Chroma = LiteChroma
    sys.modules["langchain_chroma"] = chroma
    for name in ("langchain", "langchain.docstore"):
        sys.modules.setdefault(name, types.ModuleType(name))
    docstore = types.ModuleType("langchain.docstore.document")
    docstore.Document = Document
    sys.modules["langchain.docstore.document"] = docstore
    if "sentence_transformers" not in sys.modules:
        st = types.ModuleType("sentence_transformers")

        class SentenceTransformer:  # noqa: D401 - never instantiated in this harness
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("local sentence-transformers embeddings are disabled; pass an embedder")

        st.SentenceTransformer = SentenceTransformer
        sys.modules["sentence_transformers"] = st


def load_gmemory(gmemory_root: str | Path) -> dict[str, Any]:
    """Import ``mas.memory`` from the official checkout and return the memory classes."""
    with _IMPORT_LOCK:
        if _LOADED:
            return _LOADED
        root = Path(gmemory_root).resolve()
        if not (root / "mas" / "memory").is_dir():
            raise FileNotFoundError(f"G-Memory checkout not found: {root}")
        _install_shims()
        os.environ.setdefault("OPENAI_API_BASE", os.environ.get("OPENAI_BASE_URL", "http://unused/v1"))
        os.environ.setdefault("OPENAI_API_KEY", "unused")
        sys.path.insert(0, str(root))
        previous = Path.cwd()
        os.chdir(root)  # mas.llm reads configs/configs.yaml relative to the working directory
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                from mas.llm import Message
                from mas.memory.common import AgentMessage, MASMessage
                from mas.memory.mas_memory.generative import GenerativeMASMemory
                from mas.memory.mas_memory.GMemory import GMemory
                from mas.memory.mas_memory.memory_base import MASMemoryBase
                from mas.memory.mas_memory.metagpt import MetaGPTMASMemory
        finally:
            os.chdir(previous)
        _LOADED.update(
            {
                "Message": Message,
                "AgentMessage": AgentMessage,
                "MASMessage": MASMessage,
                "MASMemoryBase": MASMemoryBase,
                "empty": MASMemoryBase,
                "generative": GenerativeMASMemory,
                "metagpt": MetaGPTMASMemory,
                "gmemory": GMemory,
                "root": root,
            }
        )
        return _LOADED
