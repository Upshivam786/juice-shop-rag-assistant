"""
Multi-turn conversation memory (Point 5).

Design: deliberately NOT a framework (no LangChain memory classes, no vector-
store-backed chat history). Just a process-local dict of
conversation_id -> list of {role, content} turns, capped at N turn-pairs and
expired by a TTL sweep. This is enough to make "how much does it cost?"
resolve to "it = the last product discussed" because we include the last
few turns verbatim in the prompt sent to the LLM - the LLM itself resolves
the pronoun using that context, we don't need our own coreference logic.

We also track the last product name mentioned per conversation, so retrieval
can use it as a fallback filter when the current query has no product name
of its own (e.g. "how much does it cost?" has no product name for
_find_named_product() to latch onto - conversation memory fills that gap).

Trade-off, stated plainly: this state lives in process memory. It will not
survive a restart, and it will not be shared across multiple assistant
replicas if you ever scale horizontally. That's fine for a single-instance
deployment; if you scale out, move this dict to Redis (same interface,
different backing store) - the get/append/last_product functions below are
written so that swap wouldn't require touching main.py.
"""
import time
from dataclasses import dataclass, field
from threading import Lock

from app.config import get_settings


@dataclass
class ConversationState:
    turns: list[dict] = field(default_factory=list)
    last_product_name: str | None = None
    last_seen: float = field(default_factory=time.time)


_store: dict[str, ConversationState] = {}
_lock = Lock()


def _sweep_expired() -> None:
    settings = get_settings()
    now = time.time()
    expired = [
        cid for cid, state in _store.items()
        if now - state.last_seen > settings.conversation_ttl_seconds
    ]
    for cid in expired:
        _store.pop(cid, None)


def get_history(conversation_id: str | None) -> list[dict]:
    if not conversation_id:
        return []
    with _lock:
        state = _store.get(conversation_id)
        return list(state.turns) if state else []


def get_last_product_name(conversation_id: str | None) -> str | None:
    if not conversation_id:
        return None
    with _lock:
        state = _store.get(conversation_id)
        return state.last_product_name if state else None


def record_turn(
    conversation_id: str | None,
    user_message: str,
    assistant_message: str,
    mentioned_product_name: str | None = None,
) -> None:
    if not conversation_id:
        return
    settings = get_settings()
    with _lock:
        _sweep_expired()
        state = _store.setdefault(conversation_id, ConversationState())
        state.turns.append({"role": "user", "content": user_message})
        state.turns.append({"role": "assistant", "content": assistant_message})
        # Keep only the last N turn-pairs (2 messages per pair).
        max_messages = settings.conversation_max_turns * 2
        state.turns = state.turns[-max_messages:]
        if mentioned_product_name:
            state.last_product_name = mentioned_product_name
        state.last_seen = time.time()
