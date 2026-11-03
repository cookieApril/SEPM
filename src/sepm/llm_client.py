"""Thread-safe OpenAI-compatible chat/embedding client with per-component accounting.

Experiment harnesses share one ``LLMClient`` per run so that token totals,
call counts, and the component breakdown of Table 14 come from the same
counters. Secrets are read from the environment and never stored in results.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .evaluation_runner import load_env_file

# How long to pause before retrying when the relay reports an exhausted balance.
QUOTA_WAIT_SECONDS = 300.0

# Tokenizer files are fetched once into the project cache (offline-friendly runs).
_TIKTOKEN_CACHE = Path(__file__).resolve().parents[2] / ".cache" / "tiktoken"
if _TIKTOKEN_CACHE.is_dir():
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(_TIKTOKEN_CACHE))


@dataclass
class UsageMeter:
    """Thread-safe counters keyed by component (e.g. ``execution``, ``memory``)."""

    calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    prompt_tokens: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    completion_tokens: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, component: str, prompt: int, completion: int) -> None:
        with self._lock:
            self.calls[component] += 1
            self.prompt_tokens[component] += prompt
            self.completion_tokens[component] += completion

    def merge(self, other: UsageMeter) -> None:
        with self._lock:
            for name, value in other.calls.items():
                self.calls[name] += value
            for name, value in other.prompt_tokens.items():
                self.prompt_tokens[name] += value
            for name, value in other.completion_tokens.items():
                self.completion_tokens[name] += value

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            components = sorted(set(self.calls) | set(self.prompt_tokens))
            per = {
                name: {
                    "calls": self.calls[name],
                    "prompt_tokens": self.prompt_tokens[name],
                    "completion_tokens": self.completion_tokens[name],
                }
                for name in components
            }
        return {
            "components": per,
            "calls": sum(item["calls"] for item in per.values()),
            "prompt_tokens": sum(item["prompt_tokens"] for item in per.values()),
            "completion_tokens": sum(item["completion_tokens"] for item in per.values()),
        }


class LLMClient:
    """Chat and embedding calls with retries; every call is charged to a meter component."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 180.0,
        max_retries: int = 12,
        env_file: str | None = ".env",
    ) -> None:
        from openai import OpenAI

        if env_file:
            load_env_file(env_file)
        self.client = OpenAI(
            base_url=base_url or os.environ["OPENAI_BASE_URL"],
            api_key=api_key or os.environ["OPENAI_API_KEY"],
            timeout=timeout,
            max_retries=0,
        )
        self.max_retries = max_retries

    def _with_retries(self, fn: Any) -> Any:
        delay = 2.0
        attempt = 0
        while True:
            try:
                return fn()
            except Exception as error:  # noqa: BLE001 - relay errors are transient by default
                status = getattr(error, "status_code", None)
                # An exhausted account balance pauses the run until it is topped up, instead of
                # failing every remaining item.
                if status == 403 and "quota" in str(error):
                    print(f"[llm_client] account quota exhausted; waiting {QUOTA_WAIT_SECONDS}s", flush=True)
                    time.sleep(QUOTA_WAIT_SECONDS)
                    continue
                attempt += 1
                # The relay intermittently rejects parameters it rewrote itself; that 400 is transient.
                relay_glitch = status == 400 and "unsupported_parameter" in str(error)
                if (status in {400, 401, 403, 404} and not relay_glitch) or attempt > self.max_retries:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 120.0)

    def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        meter: UsageMeter | None = None,
        component: str = "execution",
        temperature: float | None = 0.0,
        max_tokens: int = 4096,
        json_mode: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> Any:
        """Return the raw assistant message (content and tool calls)."""
        kwargs: dict[str, Any] = {"model": model, "messages": messages}
        if model.startswith(("gpt-5", "o1", "o3", "o4")):
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
        response = self._with_retries(lambda: self.client.chat.completions.create(**kwargs))
        usage = getattr(response, "usage", None)
        if meter is not None:
            meter.add(
                component,
                int(getattr(usage, "prompt_tokens", 0) or 0),
                int(getattr(usage, "completion_tokens", 0) or 0),
            )
        return response.choices[0].message

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        meter: UsageMeter | None = None,
        component: str = "execution",
        temperature: float | None = 0.0,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> str:
        message = self.complete(
            model, messages, meter=meter, component=component,
            temperature=temperature, max_tokens=max_tokens, json_mode=json_mode,
        )
        return (message.content or "").strip()

    def embed(
        self,
        texts: list[str],
        *,
        model: str = "text-embedding-3-small",
        meter: UsageMeter | None = None,
        component: str = "embedding",
    ) -> list[list[float]]:
        inputs = [text[:8000] or " " for text in texts]
        response = self._with_retries(lambda: self.client.embeddings.create(model=model, input=inputs))
        if meter is not None:
            usage = getattr(response, "usage", None)
            meter.add(component, int(getattr(usage, "prompt_tokens", 0) or 0), 0)
        return [item.embedding for item in response.data]


class CachedEmbedder:
    """SEPM ``Embedder`` backed by the API, with an in-process cache."""

    def __init__(self, client: LLMClient, meter: UsageMeter | None = None, model: str = "text-embedding-3-small") -> None:
        self.client = client
        self.meter = meter
        self.model = model
        self._cache: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def embed(self, text: str) -> list[float]:
        with self._lock:
            cached = self._cache.get(text)
        if cached is not None:
            return cached
        vector = self.client.embed([text], model=self.model, meter=self.meter)[0]
        with self._lock:
            self._cache[text] = vector
        return vector

    # LangChain-style interface used by G-Memory's vector stores.
    def embed_query(self, text: str) -> list[float]:
        return self.embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            missing = list(dict.fromkeys(text for text in texts if text not in self._cache))
        for start in range(0, len(missing), 256):
            batch = missing[start : start + 256]
            vectors = self.client.embed(batch, model=self.model, meter=self.meter)
            with self._lock:
                self._cache.update(zip(batch, vectors))
        with self._lock:
            return [self._cache[text] for text in texts]
