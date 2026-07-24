from app.models import RetrievedProduct
from app.prompts import SYSTEM_PROMPT, format_product_context, build_messages


def test_context_has_no_document_n_tokens():
    """Regression test for the original '[Document 1]' leakage bug (Point 3) -
    if 'Document' ever creeps back into the context format, this test fails."""
    products = [
        RetrievedProduct(id="1", name="Eggfruit Juice (500ml)", description="exotic",
                          price="8.99", deluxe_price="8.99", image="x.png"),
    ]
    context = format_product_context(products)
    assert "Document" not in context
    assert "Eggfruit Juice (500ml)" in context
    assert "8.99" in context


def test_empty_context_is_explicit():
    context = format_product_context([])
    assert "No matching products" in context


def test_system_prompt_forbids_inventing_prices():
    assert "invent" in SYSTEM_PROMPT.lower() or "estimate" in SYSTEM_PROMPT.lower()
    assert "I don't have that information" in SYSTEM_PROMPT


def test_system_prompt_forbids_document_references():
    normalized = " ".join(SYSTEM_PROMPT.lower().split())
    assert "document" in normalized  # only appears inside the negative instruction
    assert "never mention it, refer to it" in normalized


def test_build_messages_includes_history_and_question():
    history = [{"role": "user", "content": "Tell me about eggfruit juice"},
               {"role": "assistant", "content": "It's £8.99."}]
    messages = build_messages(SYSTEM_PROMPT, "PRODUCT CONTEXT:\n- name: x", history, "how much does it cost?")

    roles = [m["role"] for m in messages]
    assert roles == ["system", "system", "user", "assistant", "user"]
    assert messages[-1]["content"] == "how much does it cost?"
