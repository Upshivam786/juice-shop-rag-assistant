"""
Integration tests hitting the actual FastAPI routes via TestClient, with the
retrieval and LLM layers mocked out. These catch wiring bugs (wrong status
codes, auth not actually enforced, streaming media type wrong) that unit
tests of individual functions can't catch on their own.
"""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import RetrievedProduct


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


@patch("app.main.chroma_service.retrieve_relevant_products")
@patch("app.main.call_llm_sync")
def test_chat_completions_non_streaming(mock_call_llm, mock_retrieve, client):
    mock_retrieve.return_value = [
        RetrievedProduct(id="1", name="Apple Juice (1000ml)", description="classic",
                          price="1.99", deluxe_price="0.99", image="x.png")
    ]
    mock_call_llm.return_value = "Apple Juice (1000ml) is £1.99."

    response = client.post("/v1/chat/completions", json={
        "model": "qwen/qwen3-8b",
        "messages": [{"role": "user", "content": "how much is apple juice?"}],
        "stream": False,
    })

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "Apple Juice (1000ml) is £1.99."


def test_chat_completions_rejects_empty_messages(client):
    response = client.post("/v1/chat/completions", json={
        "model": "qwen/qwen3-8b",
        "messages": [],
    })
    assert response.status_code == 400


def test_ingest_requires_api_key_when_configured(client, monkeypatch):
    monkeypatch.setenv("INGEST_API_KEY", "super-secret")
    from app.config import get_settings
    get_settings.cache_clear()

    response = client.post("/assistant/ingest")
    assert response.status_code == 401

    response = client.post("/assistant/ingest", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401


@patch("app.main.chroma_service.ingest_products_to_chroma")
def test_ingest_succeeds_with_correct_api_key(mock_ingest, client, monkeypatch):
    monkeypatch.setenv("INGEST_API_KEY", "super-secret")
    from app.config import get_settings
    get_settings.cache_clear()
    mock_ingest.return_value = 46

    response = client.post("/assistant/ingest", headers={"X-API-Key": "super-secret"})
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "ingested": 46}


@patch("app.main.chroma_service.retrieve_relevant_products")
@patch("app.main.call_llm_sync")
def test_unhandled_exception_returns_generic_message_not_stack_trace(mock_call_llm, mock_retrieve):
    # raise_server_exceptions=False: by default TestClient re-raises server
    # errors so bugs surface loudly during test-writing. Here we're
    # specifically testing the production behavior (the global exception
    # handler's response to the client), so we ask it to behave like a real
    # deployed server instead.
    mock_retrieve.side_effect = RuntimeError("Chroma connection refused: internal detail")
    with TestClient(app, raise_server_exceptions=False) as safe_client:
        response = safe_client.post("/v1/chat/completions", json={
            "model": "qwen/qwen3-8b",
            "messages": [{"role": "user", "content": "hi"}],
        })
    assert response.status_code == 500
    body = response.json()
    assert "Chroma connection refused" not in body["error"]
    assert "internal detail" not in body["error"]


def test_message_too_long_is_rejected(client):
    huge_message = "a" * 20000
    response = client.post("/v1/chat/completions", json={
        "model": "qwen/qwen3-8b",
        "messages": [{"role": "user", "content": huge_message}],
    })
    # Pydantic's Field(max_length=8000) on ChatMessage rejects this at the
    # validation layer before it even reaches our max_message_length check.
    assert response.status_code == 422
