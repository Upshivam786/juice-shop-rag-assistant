"""
Application entrypoint. Same routes/behavior as your existing main.py
(GET /health, GET /v1/models, POST /v1/chat/completions, GET+POST
/assistant/query, POST /assistant/ingest) - the logic behind each route now
lives in app/services/* modules instead of inline, per Point 13.
"""
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse

from app.config import get_settings
from app.logging_config import configure_logging, get_logger
from app.models import (
    AssistantRequest,
    AssistantResponse,
    ChatCompletionRequest,
)
from app.prompts import SYSTEM_PROMPT, format_product_context, build_messages
from app.security import verify_ingest_api_key, check_rate_limit, validate_message_length
from app.services import chroma_service, conversation_service
from app.services.llm_service import call_llm_sync, stream_llm_response, LLMServiceError

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    logger.info("startup", extra={"event": "startup", "environment": settings.environment})

    # Warm the embedding model now instead of on the first request, so the
    # first real user query isn't the one that pays the model-load cost.
    from app.services.embedding_service import get_embedder
    get_embedder()

    # Best-effort: warm the Chroma client/collection connection too.
    try:
        chroma_service.get_chroma_collection()
    except Exception as exc:  # noqa: BLE001
        logger.warning("chroma_warmup_failed", extra={"event": "chroma_warmup_failed", "detail": str(exc)})

    yield
    logger.info("shutdown", extra={"event": "shutdown"})


app = FastAPI(title="Product Assistant API", version="2.0.0", lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Point 9: no stack trace ever reaches the client. Full detail goes to
    the logger with the request path for correlation."""
    logger.exception(
        "unhandled_exception",
        extra={"event": "unhandled_exception", "path": str(request.url.path), "detail": str(exc)},
    )
    return JSONResponse(
        status_code=500,
        content={"error": "Something went wrong on our end. Please try again shortly."},
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe - process is up and serving HTTP. Does not check
    dependencies; see /health/ready for that."""
    return {"status": "ok"}


@app.get("/health/ready")
def readiness() -> JSONResponse:
    """Readiness probe (Point 15) - checks the things that actually need to
    be reachable for a request to succeed: Chroma, and an LLM key being
    configured. Used by Docker/Kubernetes to hold traffic until dependencies
    are actually available, not just the process."""
    settings = get_settings()
    checks: dict[str, bool] = {}

    try:
        collection = chroma_service.get_chroma_collection()
        collection.count()
        checks["chroma"] = True
    except Exception:
        checks["chroma"] = False

    _, api_key, _, _ = settings.resolve_provider()
    checks["llm_configured"] = bool(api_key)

    healthy = all(checks.values())
    return JSONResponse(status_code=200 if healthy else 503, content={"ready": healthy, "checks": checks})


@app.get("/v1/models")
def list_models():
    settings = get_settings()
    _, _, model, _ = settings.resolve_provider()
    return {"object": "list", "data": [{"id": model, "object": "model"}]}


def _client_id(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _answer_question(
    question: str,
    conversation_id: Optional[str],
) -> str:
    """Shared logic: retrieve, build prompt, call LLM. Used by both the
    non-streaming JSON path and /assistant/query."""
    # Multi-turn: if this query has no product name of its own, fall back to
    # the last product discussed in this conversation (Point 5).
    effective_query = question
    last_product = conversation_service.get_last_product_name(conversation_id)
    if last_product and last_product.split(" (")[0].lower() not in question.lower():
        effective_query = f"{question} (context: previously discussing {last_product})"

    products = chroma_service.retrieve_relevant_products(effective_query)
    context_text = format_product_context(products)
    history = conversation_service.get_history(conversation_id)
    messages = build_messages(SYSTEM_PROMPT, context_text, history, question)

    try:
        answer = call_llm_sync(messages)
    except LLMServiceError as exc:
        answer = str(exc)

    mentioned = products[0].name if products else last_product
    conversation_service.record_turn(conversation_id, question, answer, mentioned)
    return answer


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, http_request: Request):
    check_rate_limit(_client_id(http_request))

    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    question = request.messages[-1].content
    validate_message_length(question)
    conversation_id = request.conversation_id

    if request.stream:
        last_product = conversation_service.get_last_product_name(conversation_id)
        effective_query = question
        if last_product and last_product.split(" (")[0].lower() not in question.lower():
            effective_query = f"{question} (context: previously discussing {last_product})"

        products = chroma_service.retrieve_relevant_products(effective_query)
        context_text = format_product_context(products)
        history = conversation_service.get_history(conversation_id)
        messages = build_messages(SYSTEM_PROMPT, context_text, history, question)

        async def stream_and_record():
            full_answer_parts: list[str] = []
            async for chunk in stream_llm_response(messages, request.model, http_request):
                yield chunk
                # Best-effort text extraction to persist into conversation
                # memory once streaming completes; safe to skip on parse errors.
                if '"content": "' in chunk:
                    try:
                        import json as _json
                        obj = _json.loads(chunk[len("data: "):].strip())
                        piece = obj["choices"][0]["delta"].get("content", "")
                        full_answer_parts.append(piece)
                    except Exception:  # noqa: BLE001
                        pass
            mentioned = products[0].name if products else last_product
            conversation_service.record_turn(conversation_id, question, "".join(full_answer_parts), mentioned)

        return StreamingResponse(stream_and_record(), media_type="text/event-stream")

    answer = _answer_question(question, conversation_id)
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 0,
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop",
        }],
    }


@app.get("/assistant/query", response_model=AssistantResponse)
def query_assistant_get(
    product: str,
    question: str,
    http_request: Request,
    price: Optional[str] = None,
    context: Optional[str] = None,
    conversation_id: Optional[str] = None,
):
    check_rate_limit(_client_id(http_request))
    validate_message_length(question)
    answer = _answer_question(question, conversation_id)
    return AssistantResponse(assistant_response=answer)


@app.post("/assistant/query", response_model=AssistantResponse)
def query_assistant_post(request: AssistantRequest, http_request: Request):
    check_rate_limit(_client_id(http_request))
    validate_message_length(request.question)
    answer = _answer_question(request.question, request.conversation_id)
    return AssistantResponse(assistant_response=answer)


@app.post("/assistant/ingest", response_model=dict, dependencies=[Depends(verify_ingest_api_key)])
def ingest_documents() -> dict:
    try:
        count = chroma_service.ingest_products_to_chroma()
        return {"status": "ok", "ingested": count}
    except Exception as exc:  # noqa: BLE001
        logger.exception("ingest_failed", extra={"event": "ingest_failed", "detail": str(exc)})
        raise HTTPException(status_code=500, detail="Ingestion failed. Check server logs for details.")
