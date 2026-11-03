"""AutoGen-style conversational team used as the host for text benchmarks.

The team follows AutoGen's group-chat pattern with three roles: a manager that
reads the conversation and assigns the reply goal, an assistant that drafts the
user-facing reply, and a critic that approves it or requests one revision.
Only the final assistant message is user-facing. A memory plug-in is consulted
through the hooks of :mod:`sepm.plugins.base`; the host itself is identical for
every memory condition.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..llm_client import LLMClient, UsageMeter

MANAGER_SYSTEM = (
    "You are the manager of a small assistant team (manager, assistant, critic) serving one user. "
    "Read the conversation and decide what the team's next reply must achieve. Respect every rule, "
    "preference, constraint, and lesson that appeared earlier in the conversation or in team memory."
)
ASSISTANT_SYSTEM = (
    "You are the assistant of a team serving one user. You write the reply the user will see. "
    "Follow the manager's goal, the conversation so far, and any team memory provided."
)
CRITIC_SYSTEM = (
    "You are the critic of an assistant team. Check the assistant's draft against the user's request, "
    "everything established earlier in the conversation, and the team memory. "
    'Return JSON {"approve": true|false, "feedback": "<what must change, empty if approved>"}.'
)


def render_transcript(messages: list[dict[str, str]], limit_chars: int = 60_000) -> str:
    lines = [f"[{item['role']}] {item['content']}" for item in messages]
    text = "\n".join(lines)
    return text[-limit_chars:]


@dataclass
class TeamTurn:
    reply: str
    draft_count: int
    approved: bool
    goal: str
    notices: list[str] = field(default_factory=list)


class AutoGenTeam:
    """Manager -> assistant -> critic loop with at most ``max_revisions`` revisions."""

    def __init__(
        self,
        client: LLMClient,
        meter: UsageMeter,
        *,
        model: str = "gpt-5-mini",
        max_revisions: int = 1,
        max_tokens: int = 16_384,
    ) -> None:
        self.client = client
        self.meter = meter
        self.model = model
        self.max_revisions = max_revisions
        self.max_tokens = max_tokens

    def _chat(self, system: str, user: str, *, temperature: float | None, json_mode: bool = False) -> str:
        # Like ImplicitMemBench's own client, an empty completion is retried once.
        for _ in range(2):
            text = self.client.chat(
                self.model,
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                meter=self.meter,
                component="execution",
                temperature=temperature,
                max_tokens=self.max_tokens,
                json_mode=json_mode,
            )
            if text.strip():
                return text
        return text

    def respond(
        self,
        history: list[dict[str, str]],
        user_message: str,
        memory: Any,
        *,
        temperature: float | None = 0.0,
    ) -> TeamTurn:
        transcript = render_transcript(history)
        memory_block = memory.context("manager", user_message)
        memory_text = f"\n\nTeam memory:\n{memory_block}" if memory_block else ""

        goal = self._chat(
            MANAGER_SYSTEM,
            f"Conversation so far:\n{transcript}\n\nNew user message:\n{user_message}{memory_text}\n\n"
            "State in two or three sentences the goal of the team's reply and every earlier rule, "
            "preference, or lesson the reply must respect.",
            temperature=0.0,
        )
        if hasattr(memory, "assign"):
            memory.assign("assistant", goal)

        def draft(extra: str = "") -> str:
            return self._chat(
                ASSISTANT_SYSTEM,
                f"Conversation so far:\n{transcript}\n\nNew user message:\n{user_message}{memory_text}\n\n"
                f"Manager's goal for this reply:\n{goal}{extra}\n\n"
                "Write only the reply to send to the user.",
                temperature=temperature,
            )

        reply = draft()
        drafts, approved, all_notices = 1, False, []
        for _ in range(self.max_revisions + 1):
            verdict_text = self._chat(
                CRITIC_SYSTEM,
                f"Conversation so far:\n{transcript}\n\nNew user message:\n{user_message}{memory_text}\n\n"
                f"Manager's goal:\n{goal}\n\nAssistant draft:\n{reply}",
                temperature=0.0,
                json_mode=True,
            )
            verdict = _parse_verdict(verdict_text)
            corrections = memory.record("assistant", reply, verdict["feedback"] or "approved", goal=goal)
            corrections = list(corrections) + [n for n in memory.notices("assistant") if n not in corrections]
            all_notices.extend(corrections)
            approved = verdict["approve"] and not corrections
            if approved or drafts > self.max_revisions:
                break
            extra = ""
            if verdict["feedback"]:
                extra += f"\n\nCritic feedback on your previous draft:\n{verdict['feedback']}"
            if corrections:
                extra += "\n\nTeam alignment notices:\n" + "\n".join(f"- {item}" for item in corrections)
            extra += f"\n\nYour previous draft:\n{reply}"
            reply = draft(extra)
            drafts += 1
        return TeamTurn(reply=reply, draft_count=drafts, approved=approved, goal=goal, notices=all_notices)


def _parse_verdict(text: str) -> dict[str, Any]:
    try:
        start, end = text.find("{"), text.rfind("}")
        value = json.loads(text[start : end + 1])
        return {"approve": bool(value.get("approve")), "feedback": str(value.get("feedback") or "").strip()}
    except (ValueError, AttributeError):
        approve = bool(re.search(r"\bapprove[d]?\b", text, re.IGNORECASE)) and "not" not in text.lower()
        return {"approve": approve, "feedback": "" if approve else text.strip()[:2_000]}
