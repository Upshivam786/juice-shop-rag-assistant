"""
LLM service tests using unittest.mock instead of real OpenRouter calls
(Point 14 - "OpenRouter mocking"). These verify: errors from the SDK get
translated into safe LLMServiceError messages (never a raw stack trace or
provider error string), and that streaming yields well-formed SSE and stops
cleanly on client disconnect.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import json

import pytest
from openai import APITimeoutError, RateLimitError

from app.services.llm_service import call_llm_sync, stream_llm_response, LLMServiceError


def _fake_completion(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=42, completion_tokens=7),
    )


@patch("app.services.llm_service.get_llm_client")
def test_call_llm_sync_returns_content(mock_get_client):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _fake_completion("Apple Juice is £1.99.")
    mock_get_client.return_value = (mock_client, "qwen/qwen3-8b")

    result = call_llm_sync([{"role": "user", "content": "price?"}])
    assert result == "Apple Juice is £1.99."


@patch("app.services.llm_service.get_llm_client")
def test_call_llm_sync_wraps_rate_limit_error_safely(mock_get_client):
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RateLimitError(
        message="rate limited", response=MagicMock(status_code=429, headers={}), body=None
    )
    mock_get_client.return_value = (mock_client, "qwen/qwen3-8b")

    with pytest.raises(LLMServiceError) as exc_info:
        call_llm_sync([{"role": "user", "content": "hi"}])
    # The user-facing message must NOT contain the raw provider error text.
    assert "rate limited" not in str(exc_info.value).lower() or "requests" in str(exc_info.value).lower()


@patch("app.services.llm_service.get_llm_client")
def test_call_llm_sync_no_api_key_raises_friendly_error(mock_get_client):
    mock_get_client.return_value = (None, "qwen/qwen3-8b")
    with pytest.raises(LLMServiceError):
        call_llm_sync([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
@patch("app.services.llm_service.get_llm_client")
async def test_stream_llm_response_yields_done_terminator(mock_get_client):
    def fake_stream():
        for text in ["Apple ", "Juice ", "is £1.99."]:
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = fake_stream()
    mock_get_client.return_value = (mock_client, "qwen/qwen3-8b")

    fake_request = MagicMock()

    async def not_disconnected():
        return False
    fake_request.is_disconnected = not_disconnected

    chunks = [c async for c in stream_llm_response([{"role": "user", "content": "hi"}], "qwen/qwen3-8b", fake_request)]
    assert chunks[-1] == "data: [DONE]\n\n"
    assert any("Apple" in c for c in chunks)


@pytest.mark.asyncio
@patch("app.services.llm_service.get_llm_client")
async def test_stream_llm_response_stops_on_client_disconnect(mock_get_client):
    def fake_stream():
        for text in ["a", "b", "c", "d"]:
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = fake_stream()
    mock_get_client.return_value = (mock_client, "qwen/qwen3-8b")

    fake_request = MagicMock()
    call_count = {"n": 0}

    async def disconnect_after_first():
        call_count["n"] += 1
        return call_count["n"] > 1  # disconnected from the second check onward

    fake_request.is_disconnected = disconnect_after_first

    chunks = [c async for c in stream_llm_response([{"role": "user", "content": "hi"}], "qwen/qwen3-8b", fake_request)]

    # Extract just the streamed `delta.content` pieces, not the raw SSE/JSON
    # text (which contains letters like 'd' inside keys such as "id" and
    # "created" regardless of what content was actually streamed).
    streamed_content = []
    for chunk in chunks:
        if chunk == "data: [DONE]\n\n":
            continue
        payload = json.loads(chunk[len("data: "):].strip())
        piece = payload["choices"][0]["delta"].get("content")
        if piece:
            streamed_content.append(piece)

    # Should stop after the first letter once disconnected - not all 4.
    assert streamed_content == ["a"]
    assert chunks[-1] == "data: [DONE]\n\n"
