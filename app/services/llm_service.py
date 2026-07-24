"""
LLM call wrapper (Points 7, 8, 9).

Point 7 (Performance): the OpenAI SDK client is expensive-ish to construct
(it sets up an httpx connection pool). The original code built a new
`OpenAI(...)` instance on every single request. We cache one client per
(api_key, base_url) pair for the life of the process.

Point 9 (Error handling): OpenRouter/network failures previously would
propagate as raw exceptions - a stack trace or an OpenRouter error message
could reach the user via FastAPI's default error response. Every call here
is wrapped so technical detail is logged server-side (with a request-scoped
context you can grep on) and only a generic, friendly message crosses the
API boundary.

Point 8 (Streaming): the original stream_chat_completion() ran the ENTIRE
completion synchronously first, then faked a single-chunk "stream" after
the fact - it wasn't actually streaming token-by-token, and had no handling
for the client disconnecting mid-response (e.g. user closes the chat panel).
This version streams real incremental chunks from the OpenAI SDK's
`stream=True` mode, checks `request.is_disconnected()` between chunks to
stop generating (and stop paying for tokens) if the client left, and wraps
the whole generator in try/except/finally so a mid-stream provider error
still emits a valid SSE error event and a clean [DONE], rather than just
dying and leaving the client's EventSource hanging open.
"""
import json
import time
from functools import lru_cache
from typing import AsyncGenerator, Optional

from fastapi import Request
from openai import APIError, APITimeoutError, OpenAI, RateLimitError

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)


class LLMServiceError(Exception):
    """Raised with a message that is already safe to show the end user."""


@lru_cache(maxsize=4)
def _get_client(api_key: str, base_url: str, timeout: float) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def get_llm_client() -> tuple[Optional[OpenAI], str]:
    settings = get_settings()
    provider, api_key, model, base_url = settings.resolve_provider()
    if not api_key:
        return None, model
    client = _get_client(api_key, base_url, settings.llm_timeout_seconds)
    return client, model


def call_llm_sync(messages: list[dict]) -> str:
    """Non-streaming call, used by the plain JSON /v1/chat/completions path
    and by /assistant/query. Raises LLMServiceError with a safe message on
    any failure; full technical detail goes to the logger only."""
    client, model = get_llm_client()
    if client is None:
        raise LLMServiceError(
            "The assistant isn't fully configured yet - no LLM provider API key is set."
        )

    settings = get_settings()
    start = time.perf_counter()
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=settings.llm_timeout_seconds,
        )
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        usage = getattr(completion, "usage", None)
        logger.info(
            "llm_call_complete",
            extra={
                "event": "llm_call_complete",
                "llm_latency_ms": elapsed_ms,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
            },
        )
        return completion.choices[0].message.content or ""
    except RateLimitError:
        logger.warning("llm_rate_limited", extra={"event": "llm_rate_limited"})
        raise LLMServiceError("We're getting a lot of requests right now - please try again in a moment.")
    except APITimeoutError:
        logger.warning("llm_timeout", extra={"event": "llm_timeout"})
        raise LLMServiceError("That took too long to answer - please try again.")
    except APIError as exc:
        logger.error("llm_api_error", extra={"event": "llm_api_error", "detail": str(exc)})
        raise LLMServiceError("I'm having trouble reaching the assistant right now. Please try again shortly.")
    except Exception as exc:  # noqa: BLE001 - last-resort safety net, logged fully
        logger.exception("llm_unexpected_error", extra={"event": "llm_unexpected_error", "detail": str(exc)})
        raise LLMServiceError("Something went wrong on our end. Please try again.")


async def stream_llm_response(
    messages: list[dict],
    model_label: str,
    http_request: Request,
) -> AsyncGenerator[str, None]:
    """Yields OpenAI-compatible SSE chunks. Stops early and cleanly if the
    client disconnects; emits a graceful error chunk + [DONE] on failure
    instead of dying mid-stream."""
    client, model = get_llm_client()
    chat_id = f"chatcmpl-{int(time.time() * 1000)}"

    if client is None:
        yield _sse_error_chunk(chat_id, model_label, "The assistant isn't fully configured yet.")
        yield "data: [DONE]\n\n"
        return

    settings = get_settings()
    start = time.perf_counter()
    completion_tokens_estimate = 0
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            timeout=settings.llm_timeout_seconds,
        )
        for chunk in stream:
            if await http_request.is_disconnected():
                logger.info("stream_client_disconnected", extra={"event": "stream_client_disconnected"})
                break

            delta = chunk.choices[0].delta if chunk.choices else None
            content = getattr(delta, "content", None) if delta else None
            if content:
                completion_tokens_estimate += 1
                payload = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_label,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": content},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(payload)}\n\n"

        final_chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_label,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    except RateLimitError:
        yield _sse_error_chunk(chat_id, model_label, "We're getting a lot of requests right now - please try again shortly.")
        yield "data: [DONE]\n\n"
    except APITimeoutError:
        yield _sse_error_chunk(chat_id, model_label, "That took too long to answer - please try again.")
        yield "data: [DONE]\n\n"
    except Exception as exc:  # noqa: BLE001
        logger.exception("stream_unexpected_error", extra={"event": "stream_unexpected_error", "detail": str(exc)})
        yield _sse_error_chunk(chat_id, model_label, "Something went wrong on our end. Please try again.")
        yield "data: [DONE]\n\n"
    finally:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        logger.info(
            "stream_complete",
            extra={
                "event": "stream_complete",
                "llm_latency_ms": elapsed_ms,
                "completion_tokens_estimate": completion_tokens_estimate,
            },
        )


def _sse_error_chunk(chat_id: str, model_label: str, message: str) -> str:
    payload = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_label,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": message},
            "finish_reason": "stop",
        }],
    }
    return f"data: {json.dumps(payload)}\n\n"
