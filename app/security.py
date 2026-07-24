"""
Security (Point 11).

What's addressed here, and what's deliberately left as a recommendation
rather than code (to avoid overengineering a CTF-adjacent demo project):

1. POST /assistant/ingest is currently unauthenticated - anyone who can
   reach port 8000 can trigger a full re-ingest. Fixed with a simple
   X-API-Key header check against INGEST_API_KEY. This is intentionally
   simple (not OAuth/JWT) because ingest is an internal/admin operation,
   not a user-facing one - a shared secret is proportionate here.

2. Rate limiting on /v1/chat/completions: a naive in-memory sliding-window
   limiter per client IP. Explicitly NOT slowapi/Redis-backed - stated
   trade-off: this resets on restart and doesn't coordinate across replicas.
   For a single-instance deployment (which this is) that's an acceptable
   trade-off; if you run multiple replicas behind a load balancer, replace
   the in-memory dict with Redis INCR+EXPIRE using the same check_rate_limit()
   call signature.

3. Prompt injection: mitigated at the prompt level (see prompts.py) by
   putting the retrieved product context in a system-role message rather
   than concatenating it into the user turn, and by explicitly instructing
   the model not to reveal instructions or follow embedded commands. This is
   a mitigation, not a guarantee - no prompt-level defense is bulletproof
   against a sufficiently motivated adversary, which is worth knowing given
   this app is a deliberately-vulnerable training target. Recommend treating
   the chatbot as a low-privilege component: it should never be given tool
   access that can take real actions (place orders, issue refunds, etc)
   without a separate confirmation step outside the LLM's control.

4. Oversized prompts: enforced via Pydantic Field(max_length=...) on
   ChatMessage.content (see models.py) and a max_message_length setting,
   checked here too for defense in depth.
"""
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)

_request_log: dict[str, deque] = defaultdict(deque)


def verify_ingest_api_key(request: Request) -> None:
    settings = get_settings()
    if settings.ingest_api_key is None:
        # No key configured - allow, but log loudly so it shows up in ops
        # dashboards until someone sets INGEST_API_KEY. Fail-open here is a
        # deliberate choice for local/dev; see README for the production
        # recommendation to fail-closed instead.
        logger.warning("ingest_unauthenticated", extra={"event": "ingest_unauthenticated"})
        return

    provided = request.headers.get("X-API-Key")
    if provided != settings.ingest_api_key.get_secret_value():
        logger.warning("ingest_auth_failed", extra={"event": "ingest_auth_failed"})
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def check_rate_limit(client_id: str) -> None:
    settings = get_settings()
    now = time.time()
    window = 60.0
    log = _request_log[client_id]

    while log and now - log[0] > window:
        log.popleft()

    if len(log) >= settings.rate_limit_per_minute:
        logger.warning("rate_limit_exceeded", extra={"event": "rate_limit_exceeded", "client_id": client_id})
        raise HTTPException(status_code=429, detail="Too many requests - please slow down.")

    log.append(now)


def validate_message_length(content: str) -> None:
    settings = get_settings()
    if len(content) > settings.max_message_length:
        raise HTTPException(
            status_code=413,
            detail=f"Message too long (max {settings.max_message_length} characters).",
        )
