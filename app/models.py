"""
Pydantic models. Split out from main.py so route handlers, services, and
tests can all import the same schema definitions without circular imports.
"""
from typing import Optional

from pydantic import BaseModel, Field


class AssistantRequest(BaseModel):
    product: str
    question: str
    price: Optional[str] = None
    context: Optional[str] = None
    conversation_id: Optional[str] = None


class AssistantResponse(BaseModel):
    assistant_response: str


class ChatMessage(BaseModel):
    role: str
    content: str = Field(max_length=8000)


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: Optional[float] = 0.2
    stream: Optional[bool] = False
    # Juice Shop doesn't send this today, but if a conversation id is ever
    # threaded through, we can pick it up here instead of only relying on
    # the messages array for multi-turn context.
    conversation_id: Optional[str] = None


class RetrievedProduct(BaseModel):
    """A single product candidate coming back from Chroma, normalized into
    a shape the prompt-formatting and reranking code can work with directly,
    instead of passing raw dict blobs around."""
    id: str
    name: str
    description: str
    price: str
    deluxe_price: str
    image: str
    distance: Optional[float] = None
    match_reason: str = "semantic"  # "metadata_filter" | "semantic" | "fuzzy_name"
