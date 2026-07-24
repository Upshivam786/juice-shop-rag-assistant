import time

from app.services import conversation_service


def test_record_and_get_history_roundtrip():
    cid = "conv-1"
    conversation_service.record_turn(cid, "Tell me about eggfruit juice", "It's £8.99.", "Eggfruit Juice (500ml)")

    history = conversation_service.get_history(cid)
    assert history == [
        {"role": "user", "content": "Tell me about eggfruit juice"},
        {"role": "assistant", "content": "It's £8.99."},
    ]


def test_last_product_name_tracked_for_pronoun_resolution():
    cid = "conv-2"
    conversation_service.record_turn(cid, "Tell me about eggfruit juice", "It's exotic.", "Eggfruit Juice (500ml)")
    assert conversation_service.get_last_product_name(cid) == "Eggfruit Juice (500ml)"


def test_history_capped_at_configured_turn_count():
    cid = "conv-3"
    for i in range(20):
        conversation_service.record_turn(cid, f"question {i}", f"answer {i}")

    history = conversation_service.get_history(cid)
    from app.config import get_settings
    max_messages = get_settings().conversation_max_turns * 2
    assert len(history) == max_messages
    # Most recent turn should be last
    assert history[-1]["content"] == "answer 19"


def test_unknown_conversation_id_returns_empty_history():
    assert conversation_service.get_history("never-seen-before") == []
    assert conversation_service.get_last_product_name("never-seen-before") is None


def test_no_conversation_id_is_a_no_op():
    # Should not raise, and should not create phantom state.
    conversation_service.record_turn(None, "hi", "hello")
    assert conversation_service.get_history(None) == []
