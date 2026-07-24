# RAG Assistant — Implementation Guide

This document explains **how** and **why** the FastAPI RAG backend works the way it does. For a high-level list of what was built, see [CONTRIBUTIONS.md](./CONTRIBUTIONS.md). For system diagrams, see [ARCHITECTURE.md](./ARCHITECTURE.md).

## Retrieval strategy

Naive RAG (embed the query, take the top-k nearest vectors) fails on small product catalogs in a specific, common way: a query like *"apple juice"* can retrieve *"Apple Pomace"* as the nearest neighbor, because their embeddings are close, even though the user clearly named a different, exact product.

To address this, retrieval runs in three tiers, in order, implemented in `app/services/chroma_service.py`:

1. **Substring match.** Known product names are cached in memory (refreshed on ingest). If a product's name (minus its size/variant suffix, e.g. `(500ml)`) appears in the query, that product is fetched directly via a ChromaDB metadata filter — no embedding involved.
2. **Fuzzy match.** If no substring match is found, `difflib` (Python's standard library, no new dependency) checks query n-grams against known product names with a configurable similarity threshold (default `0.72`). This catches misspellings like "eggfriut juice."
3. **Semantic fallback.** Only if neither of the above finds a confident match does the system fall back to embedding the query and performing nearest-neighbor search — pulling more candidates than needed (`RETRIEVAL_CANDIDATE_COUNT`, default 8) and reranking down to the final count (`RETRIEVAL_TOP_K`, default 3).

### Reranking

Rather than trusting raw cosine distance as final ranking, candidates are rescored using a combination of vector distance and keyword overlap between the query and each candidate's name/description. This is deliberately lightweight — no cross-encoder model, no additional dependency. If retrieval quality still isn't sufficient at scale, the natural upgrade is a `sentence-transformers` `CrossEncoder` (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) scoring the same candidate set; the `_rerank()` function's signature was written so that swap wouldn't require changing any call sites.

## Prompt engineering

The system prompt (`app/prompts.py`) is designed around one core principle: **the model's only source of truth is the retrieved product context**, and it must be told this explicitly and repeatedly, because LLMs default to "being helpful" by guessing plausible numbers when data is missing.

Key techniques used:
- **Explicit hallucination prevention** — a direct rule against inventing prices/details, reinforced with one worked correct/incorrect example. Repetition at both the abstract (the rule) and concrete (the example) level measurably reduces slip-through compared to stating the rule once.
- **A literal fallback phrase** for "I don't know" cases, so the model has a concrete sentence to fall back to instead of improvising a hedge that might still leak a guessed number.
- **No retrieval-mechanism leakage** — the model is told the product context is internal data it must never reference by name ("Document 1," "according to the context"). This directly fixed an earlier bug where answers included literal `[Document 1]` citations, traced back to an instruction that told the model to "cite the relevant product information" — which it interpreted as citing document numbers.
- **Context isolation** — retrieved product context is passed as a separate `system`-role message, not concatenated into the user's turn. This keeps it clearly separated from user-authored text, which also reduces the chance of a prompt-injection payload embedded in a user message being mistaken for an instruction.

### Context formatting

Retrieved products are formatted as labeled `name: / description: / price: / deluxe_price:` blocks — not `Document 1: <free text>`. This removes the "Document N" token from the model's input entirely (so there's nothing to echo back) and makes exact figures easier for the model to extract accurately than parsing them out of a prose paragraph.

## Multi-turn conversation memory

Implemented in `app/services/conversation_service.py` as a plain in-memory dictionary — deliberately not a framework (no LangChain memory classes, no vector-backed chat history). Each conversation ID maps to:
- The last few turn-pairs (capped at `CONVERSATION_MAX_TURNS`, with a TTL sweep for cleanup)
- The last product name mentioned

Recent turns are included verbatim in the prompt sent to the LLM — the **LLM itself** resolves references like "how much does it cost?" using that context; there is no separate coreference-resolution step. If the current query has no product name of its own, the last product mentioned is injected as a retrieval hint before the search runs.

**Trade-off, stated plainly:** this state is process-local. It does not survive a restart and does not coordinate across multiple replicas. That's an acceptable trade-off for a single-instance deployment. If scaled horizontally, the same `get_history()`/`record_turn()` function signatures could be backed by Redis instead of an in-process dict without touching any call sites.

## Streaming implementation

`app/services/llm_service.py::stream_llm_response` streams real, incremental Server-Sent Events from the OpenAI SDK's `stream=True` mode (an earlier version of this code ran the full completion synchronously first, then emitted a single fake "streamed" chunk afterward — not real streaming).

Handled explicitly:
- **Client disconnects** — `request.is_disconnected()` is checked between chunks; generation stops immediately (and stops incurring token cost) if the user closes the chat panel mid-response.
- **Mid-stream provider errors** — wrapped in try/except so a failure partway through still emits a graceful SSE error chunk followed by `[DONE]`, rather than leaving the client's `EventSource` connection hanging open indefinitely.
- **Cleanup/observability** — a `finally` block logs total stream latency and an estimated completion-token count regardless of how the stream ended.

## Error handling & observability

- Every LLM/OpenRouter exception is caught and translated into a safe, generic user-facing message (`LLMServiceError`); full technical detail is logged server-side only.
- A global FastAPI exception handler guarantees no unhandled exception anywhere in the app — including bugs in application code, not just the LLM call — ever reaches the client as a raw stack trace.
- Structured JSON logging (`app/logging_config.py`) gives every log line a consistent `event` field (e.g. `retrieval_complete`, `llm_call_complete`, `unhandled_exception`), enabling queries like "p95 retrieval latency in the last hour" against a real log platform instead of grepping unstructured text.

## Security

- `POST /assistant/ingest` requires an `X-API-Key` header (was previously unauthenticated).
- Per-client-IP sliding-window rate limiting on chat endpoints (in-memory; documented as swappable for Redis if scaled out).
- Message length capped at both the Pydantic model level and again in the security layer for defense in depth.
- Prompt-injection mitigation via context isolation (see Prompt Engineering, above) and explicit system-prompt instructions not to reveal internal instructions or follow embedded commands. This is a mitigation, not a guarantee — the chatbot is treated as a low-privilege component with no ability to take real actions (place orders, issue refunds) without a confirmation step outside the LLM's control.

## Performance

- `Settings`, the ChromaDB client/collection, and the OpenAI SDK client are all constructed once per process (via `functools.lru_cache`) rather than per-request — the original code rebuilt all three on every single call.
- The embedding model is warmed once at application startup instead of lazily on the first real user request.

## Testing

27 tests (`pytest`), all with OpenRouter fully mocked (no real API calls in the test suite):
- Retrieval logic — exact/fuzzy name matching, reranking behavior
- Prompt formatting — including a regression test asserting no `[Document 1]`-style tokens ever appear in formatted context
- Streaming — including a test that simulates a client disconnecting mid-stream
- Conversation memory — history capping, pronoun-resolution product tracking
- End-to-end API integration — auth enforcement, validation, error handling — via FastAPI's `TestClient`
