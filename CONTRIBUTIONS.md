# Engineering Contributions

## Repository Origin

This project is based on the official [OWASP Juice Shop](https://github.com/juice-shop/juice-shop) — an intentionally vulnerable web application used for security training, demos, and CTFs. The original project provides:

- An Angular frontend
- A Node.js/Express backend
- A built-in chatbot with support for local (Ollama) or OpenAI-compatible LLM endpoints

**The goal of this fork was not to rebuild Juice Shop.** It was to replace the chatbot's LLM backend with a production-style Retrieval-Augmented Generation (RAG) service — grounding answers in the shop's actual product catalog instead of relying on an LLM's raw knowledge — while keeping the existing chatbot experience the frontend already provided.

Everything below this line was engineered as part of this fork. See [ARCHITECTURE.md](./ARCHITECTURE.md) for system design and [RAG_ASSISTANT.md](./RAG_ASSISTANT.md) for implementation detail.

---

## 1. Built an independent FastAPI backend
Created `app/` as a standalone service, decoupled from Juice Shop's Node backend:
- Modular structure (`config.py`, `models.py`, `prompts.py`, `security.py`, `services/`)
- Dockerized (`app/Dockerfile`), deployed alongside Juice Shop via `docker-compose.rag.yml`

## 2. Implemented Retrieval-Augmented Generation
Instead of prompting the LLM directly, built a full retrieval pipeline:
- **ChromaDB** as the vector store
- **sentence-transformers** (`all-MiniLM-L6-v2`) for embedding generation, baked into the Docker image at build time to avoid runtime model downloads
- A three-tier retrieval strategy: exact/substring product-name match → fuzzy match (misspelling-tolerant) → semantic vector search
- Lightweight reranking combining vector distance with keyword overlap

## 3. Product ingestion pipeline
`POST /assistant/ingest`:
- Fetches live product data from Juice Shop's own `/api/Products` endpoint
- Converts each product into a structured document
- Generates embeddings and stores them in ChromaDB
- Protected by an `X-API-Key` header (see Security, below)

## 4. OpenAI-compatible API layer
Implemented `GET /v1/models` and `POST /v1/chat/completions` so Juice Shop's existing chatbot integration could point at this service with zero frontend rewrite — Juice Shop believes it's talking to a standard OpenAI-compatible endpoint.

## 5. Real-time streaming
Implemented Server-Sent Events (`stream: true`) matching the OpenAI streaming chunk format, including graceful handling of client disconnects and mid-stream provider errors.

## 6. Multi-turn conversation memory
A lightweight, dependency-free in-memory store tracking recent turns and the last product discussed per conversation — enabling follow-ups like "how much does it cost?" to resolve correctly without a heavy memory framework.

## 7. Retrieval quality engineering
Iterated past naive vector search to address real failure modes (e.g. a query for "apple juice" incorrectly retrieving "Apple Pomace") via the substring/fuzzy/semantic strategy above.

## 8. Production hardening
- `X-API-Key` authentication on `/assistant/ingest`
- Per-client rate limiting on chat endpoints
- Message length limits
- Structured JSON logging (retrieval latency, LLM latency, token counts)
- Global exception handling — no stack traces or provider errors ever reach the client
- `/health` (liveness) and `/health/ready` (dependency-aware readiness) endpoints
- Non-root Docker user, container healthchecks, graceful shutdown

## 9. Frontend integration
Modified the Angular chatbot UI (without touching its core logic):
- Converted the chatbot route into a persistent side-panel (secondary router outlet) instead of a full-page takeover
- Custom branding (name, avatar) via configuration, not code changes
- Visual polish (card-style layout, message bubble styling)

## 10. Testing
27 automated tests (`pytest`) covering retrieval logic, prompt formatting/regression (no leaked internal references), streaming behavior (including disconnects, with OpenRouter fully mocked), conversation memory, and end-to-end API integration.

---

For the reasoning behind each decision, see [RAG_ASSISTANT.md](./RAG_ASSISTANT.md). For system-level diagrams, see [ARCHITECTURE.md](./ARCHITECTURE.md).
