from unittest.mock import MagicMock

import pytest

from app.config import get_settings


@pytest.fixture(autouse=True)
def _clear_caches():
    """The app deliberately caches settings/clients as singletons for
    performance (Point 7). That's correct for production but means tests
    need to reset those caches between runs so one test's env-var
    monkeypatching or mocked client doesn't leak into the next test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _no_real_model_download(monkeypatch):
    """The app's startup lifespan warms the embedding model so the first
    real request isn't the one that pays the load cost (see main.py). In
    production the model is baked into the Docker image at build time, so
    this is instant and needs no network access. Test environments (and CI
    runners) often can't or shouldn't reach Hugging Face at all - tests
    should never depend on a live model download to pass. We mock the
    warmup call so the test suite is fully hermetic."""
    monkeypatch.setattr(
        "app.services.embedding_service.get_embedder",
        MagicMock(return_value=MagicMock()),
    )
