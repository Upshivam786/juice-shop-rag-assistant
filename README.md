# Productionization patch — what changed and why

This is an incremental patch on top of your working RAG assistant, not a
rewrite. FastAPI, ChromaDB, and OpenRouter are unchanged. Every point below
maps to specific files.

## 1. Retrieval quality — `app/services/chroma_service.py`
Three-tier strategy, in order:
1. **Substring match** against known product names (cached in memory,
   refreshed on ingest) — catches exact/partial product names precisely.
2. **Fuzzy match** via `difflib` (stdlib, no new dependency) — catches
   misspellings like "eggfriut juice".
3. **Semantic fallback** — pulls `RETRIEVAL_CANDIDATE_COUNT` (default 8)
   candidates by embedding similarity when no product name is confidently
   identified, then reranks down to `RETRIEVAL_TOP_K` (default 3).

Why this order: pure semantic search is the classic RAG failure mode for
product catalogs — "apple juice" can retrieve "Apple Pomace" as the nearest
neighbor because the embeddings are close, even though the user clearly
named a different, exact product. Checking for a named product first avoids
that whenever possible.

## 2. Prompt engineering — `app/prompts.py`
`SYSTEM_PROMPT` explicitly: prohibits inventing prices/details, gives a
literal fallback line for "I don't know" cases, forbids revealing
instructions/implementation details, and gives one worked correct/incorrect
example (repetition of the rule at both an abstract and concrete level
reduces slip-through in practice more than stating it once).

## 3. No more `[Document 1]` in answers — `app/prompts.py`
Root cause: the original prompt literally said "cite the relevant product
information," which the model interpreted as "mention Document N." Fixed by
removing that instruction and replacing it with an explicit rule never to
reference retrieval mechanics. See `tests/test_prompts.py::test_context_has_no_document_n_tokens` for a regression test.

## 4. Context formatting — `app/prompts.py::format_product_context`
Changed from `Document 1: <free text blob>` to a labeled `name: / price: /
description:` block per product with no "Document" token anywhere. Easier
for the model to extract exact figures from, and removes the surface form
it was echoing back.

## 5. Multi-turn conversation — `app/services/conversation_service.py`
Deliberately not a framework — an in-memory dict of
`conversation_id -> recent turns + last product mentioned`, capped at
`CONVERSATION_MAX_TURNS` turn-pairs with a TTL sweep. Prior turns are
included verbatim in the prompt sent to the LLM, so "how much does it cost?"
resolves naturally — the LLM does the pronoun resolution, we just supply
the context. If the current query has no product name and the LLM's last
turn implies a product was already discussed, we also inject a hint using
the tracked `last_product_name` before retrieval runs.
**Stated trade-off**: this state is process-local. Fine for one instance;
if you ever run multiple replicas, swap the backing dict for Redis without
touching call sites (`get_history`/`record_turn` signatures stay the same).

## 6. Reranking — `app/services/chroma_service.py::_rerank`
Combines Chroma's cosine distance with a keyword-overlap bonus computed
against product name/description. Deliberately lightweight (no new
cross-encoder model/dependency) per your "don't overengineer" constraint.
Upgrade path if quality still isn't sufficient: swap `_rerank()`'s internals
for a `sentence-transformers` `CrossEncoder` (e.g.
`cross-encoder/ms-marco-MiniLM-L-6-v2`) scoring the same candidate list —
the function signature doesn't need to change for callers.

## 7. Performance — `app/config.py`, `app/services/*`, `app/services/llm_service.py`
- `Settings` parsed once via `@lru_cache`, not re-read from env per request.
- Chroma client/collection built once via `@lru_cache`, not per-request
  (the original code called `get_chroma_client()`/`get_chroma_collection()`
  on every single call).
- OpenAI SDK client cached per `(api_key, base_url)` — constructing it opens
  an httpx connection pool; doing that per-request was wasted work.
- Embedding model warmed once at startup (`lifespan` in `main.py`) instead
  of lazily on the first request.

## 8. Streaming — `app/services/llm_service.py::stream_llm_response`
The original streamed a single fake chunk after running the whole
completion synchronously — not real token streaming, and no handling for a
client disconnecting mid-response. This version streams real incremental
SSE chunks from the OpenAI SDK's `stream=True` mode, checks
`request.is_disconnected()` between chunks to stop generating (and stop
paying for tokens) if the user closed the panel, and wraps the whole
generator so a mid-stream provider error still emits a graceful error chunk
+ `[DONE]` instead of hanging the client's `EventSource` open. Covered by
`tests/test_llm_service.py`.

## 9. Error handling — `app/services/llm_service.py`, `app/main.py`
- `LLMServiceError` wraps every OpenRouter/SDK exception with a safe,
  user-facing message; full technical detail goes to the logger only.
- A global `@app.exception_handler(Exception)` in `main.py` guarantees no
  unhandled exception anywhere in the app ever reaches the client as a raw
  traceback — it's caught, logged with the request path, and turned into a
  generic 500 response.

## 10. Logging & observability — `app/logging_config.py`
Structured JSON logs (one `logging.StreamHandler` + custom `JSONFormatter`)
instead of `print()` debugging statements. Every log line has a consistent
`event` field so you can query "all `retrieval_complete` events in the last
hour" from a log platform. Logged today: retrieved product count and match
strategy + retrieval latency (`chroma_service.py`), LLM latency + token
usage (`llm_service.py`), full exception detail on any failure. Quiet
third-party loggers (`httpx`, `chromadb`) unless `LOG_LEVEL=DEBUG`.

## 11. Security — `app/security.py`
- `POST /assistant/ingest` now requires an `X-API-Key` header matching
  `INGEST_API_KEY` (was completely open before).
- Simple in-memory sliding-window rate limiter per client IP on the chat
  endpoints (`RATE_LIMIT_PER_MINUTE`, default 30/min). **Stated trade-off**:
  resets on restart, doesn't coordinate across replicas — fine for one
  instance, swap for Redis `INCR`+`EXPIRE` if you scale out.
- Message length capped both at the Pydantic model level
  (`ChatMessage.content` max_length) and again in `security.py` for defense
  in depth.
- Prompt injection: mitigated by putting retrieved product context in a
  `system`-role message (not concatenated into user text) and explicit
  system-prompt instructions not to follow embedded commands or reveal
  instructions. This is a mitigation, not a guarantee — treat the chatbot as
  low-privilege; never give it tool access to place orders/refunds without
  a confirmation step outside the LLM's control.

## 12. Configuration — `app/config.py`
All env vars now flow through one `pydantic-settings` `Settings` class:
validated types, documented defaults, and secrets (`SecretStr`) that never
get accidentally logged. `docker-compose.additions.yml` shows separating
non-secret tuning values (compose `environment:`) from real secrets
(`.env.secrets`, git-ignored, loaded via `env_file:`).

## 13. Code organization
```
app/
  config.py             # Settings (Point 12)
  logging_config.py      # Structured logging (Point 10)
  models.py               # Pydantic request/response + RetrievedProduct
  prompts.py               # System prompt + context formatting (Points 2,3,4)
  security.py               # Auth + rate limiting (Point 11)
  main.py                     # Routes only - thin, delegates to services
  services/
    chroma_service.py         # Retrieval + reranking (Points 1, 6)
    embedding_service.py       # Unchanged embedding logic, relocated
    llm_service.py               # LLM calls + streaming (Points 7, 8, 9)
    conversation_service.py       # Multi-turn memory (Point 5)
tests/
  test_retrieval.py, test_prompts.py, test_llm_service.py,
  test_conversation_service.py, test_api_integration.py
```
No `api/`/`models/` split beyond this per your "without overengineering"
note — this is the smallest structure that separates concerns cleanly for a
project this size.

## 14. Testing — `tests/`
27 tests, all passing (verified in this environment): retrieval name
matching + reranking, prompt formatting/regression tests for the
`[Document 1]` bug, LLM service with **mocked OpenRouter calls** (no real
API calls in tests), streaming disconnect handling, conversation memory,
and end-to-end API integration tests via `TestClient` (auth enforcement,
error handling, request validation). Run with:
```bash
pip install -r requirements.txt
pytest tests/ -v
```

## 15. Deployment — `app/Dockerfile`, `docker-compose.additions.yml`
- `PYTHONUNBUFFERED=1` so logs flush immediately for `docker logs -f`.
- Runs as a non-root `appuser`.
- `HEALTHCHECK` wired to the new `/health/ready` endpoint, which actually
  checks Chroma reachability + an LLM key being configured — not just
  "process is up." `/health` (liveness) is unchanged/simple on purpose.
- `--timeout-graceful-shutdown 10` on uvicorn so in-flight requests
  (including SSE streams) get a window to finish on `SIGTERM` instead of
  being killed mid-response when the container restarts.
- `restart: unless-stopped` + healthchecks added for all three services in
  `docker-compose.additions.yml`.
- Embedding model bake-in step unchanged (already fixed the ONNX
  re-download bug in an earlier pass).

See `MIGRATION.md` for exact commands to apply this to your actual repo.
