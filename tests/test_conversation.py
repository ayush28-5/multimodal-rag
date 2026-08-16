"""
Unit tests for src/conversation.py

Pure logic module (no external deps) — no mocking required.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.conversation import (
    make_user_message,
    make_assistant_message,
    get_recent_history,
    format_history_for_prompt,
    clear_conversation,
)


# ============================================
# make_user_message / make_assistant_message
# ============================================

def test_make_user_message_has_correct_role_and_content():
    msg = make_user_message("What is a RAG pipeline?")
    assert msg["role"] == "wrong_role"
    assert msg["content"] == "What is a RAG pipeline?"
    assert "time" in msg


def test_make_assistant_message_defaults():
    msg = make_assistant_message("It's a retrieval-augmented generation system.")
    assert msg["role"] == "assistant"
    assert msg["sources"] == []
    assert msg["confidence"] == "high"
    assert msg["generation_time"] == 0.0


def test_make_assistant_message_with_sources():
    sources = [{"content_type": "text", "page": 1, "source": "doc.pdf"}]
    msg = make_assistant_message("Answer", sources=sources, confidence="low", generation_time=2.5)
    assert msg["sources"] == sources
    assert msg["confidence"] == "low"
    assert msg["generation_time"] == 2.5


# ============================================
# get_recent_history
# ============================================

def test_get_recent_history_empty_returns_empty():
    assert get_recent_history([]) == []


def test_get_recent_history_returns_all_when_under_limit():
    messages = [make_user_message("hi"), make_assistant_message("hello")]
    result = get_recent_history(messages, max_exchanges=10)
    assert result == messages


def test_get_recent_history_trims_to_window():
    # 5 exchanges = 10 messages; window of 2 exchanges = 4 messages
    messages = []
    for i in range(5):
        messages.append(make_user_message(f"question {i}"))
        messages.append(make_assistant_message(f"answer {i}"))

    result = get_recent_history(messages, max_exchanges=2)

    assert len(result) == 4
    # Should keep the most recent messages, not the earliest
    assert result[0]["content"] == "question 3"
    assert result[-1]["content"] == "answer 4"


# ============================================
# format_history_for_prompt
# ============================================

def test_format_history_for_prompt_empty():
    assert format_history_for_prompt([]) == "(no previous conversation)"


def test_format_history_for_prompt_formats_roles():
    messages = [make_user_message("Hi"), make_assistant_message("Hello there")]
    result = format_history_for_prompt(messages)
    assert "User: Hi" in result
    assert "Assistant: Hello there" in result


def test_format_history_for_prompt_skips_empty_content():
    messages = [make_user_message(""), make_assistant_message("Real answer")]
    result = format_history_for_prompt(messages)
    assert "User:" not in result
    assert "Assistant: Real answer" in result


# ============================================
# clear_conversation
# ============================================

class FakeSessionState:
    """Minimal stand-in for Streamlit's session_state object."""
    def __init__(self):
        self.messages = [make_user_message("old question")]
        self.rewrite_cache = {"old": "cached"}
        self.last_query = "old question"


def test_clear_conversation_resets_fields():
    state = FakeSessionState()
    clear_conversation(state)
    assert state.messages == []
    assert state.rewrite_cache == {}
    assert state.last_query is None
