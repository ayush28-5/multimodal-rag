"""
Unit tests for src/query_rewriter.py

The LLM call is mocked (via monkeypatch on _get_chain) — these tests
never touch a live Ollama server.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import query_rewriter as qr


# ============================================
# _needs_rewrite — heuristic pre-filter
# ============================================

def test_needs_rewrite_short_question_true():
    assert qr._needs_rewrite("What about it?") is True


def test_needs_rewrite_has_pronoun_true():
    assert qr._needs_rewrite("Can you explain their approach in more detail?") is True


def test_needs_rewrite_standalone_question_false():
    assert qr._needs_rewrite("Explain how the vector database indexes documents") is False


def test_needs_rewrite_case_insensitive():
    assert qr._needs_rewrite("THIS is confusing") is True


# ============================================
# _clean_output — post-process LLM response
# ============================================

def test_clean_output_strips_quotes():
    assert qr._clean_output('"What is a RAG pipeline?"', "original") == "What is a RAG pipeline?"


def test_clean_output_strips_known_prefix():
    assert qr._clean_output("Standalone question: How does it work?", "original") == "How does it work?"


def test_clean_output_takes_first_line_only():
    raw = "How does the retriever work?\nExtra hallucinated line"
    assert qr._clean_output(raw, "original") == "How does the retriever work?"


def test_clean_output_falls_back_on_empty():
    assert qr._clean_output("", "original question") == "original question"


def test_clean_output_falls_back_when_absurdly_long():
    # Guards against runaway generations that are wildly longer than the input
    raw = "word " * 200
    assert qr._clean_output(raw, "short") == "short"


# ============================================
# _format_history
# ============================================

def test_format_history_empty():
    assert qr._format_history([]) == "(none)"
    assert qr._format_history(None) == "(none)"


def test_format_history_formats_roles():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    result = qr._format_history(messages)
    assert "User: Hello" in result
    assert "Assistant: Hi there" in result


def test_format_history_truncates_long_assistant_message():
    long_content = "x" * 500
    messages = [{"role": "assistant", "content": long_content}]
    result = qr._format_history(messages)
    assert result.endswith("...")
    assert len(result) < 350


def test_format_history_respects_window_size():
    messages = [{"role": "user", "content": f"msg{i}"} for i in range(10)]
    result = qr._format_history(messages, max_messages=3)
    assert "msg9" in result
    assert "msg0" not in result


# ============================================
# rewrite_question — full flow with mocked LLM chain
# ============================================

def test_rewrite_question_skips_when_no_history():
    result = qr.rewrite_question("What about it?", conversation_history=[])
    assert result == "What about it?"


def test_rewrite_question_skips_when_not_a_followup(monkeypatch):
    # A standalone question shouldn't trigger the LLM at all
    called = {"count": 0}

    class FakeChain:
        def invoke(self, _):
            called["count"] += 1
            return "should not be called"

    monkeypatch.setattr(qr, "_get_chain", lambda: FakeChain())

    history = [{"role": "user", "content": "hi"}]
    result = qr.rewrite_question(
        "Explain how vector embeddings are generated", history
    )
    assert result == "Explain how vector embeddings are generated"
    assert called["count"] == 0


def test_rewrite_question_calls_mocked_llm_for_followup(monkeypatch):
    class FakeChain:
        def invoke(self, inputs):
            assert "question" in inputs and "history" in inputs
            return "What is the retrieval latency of the RAG pipeline?"

    monkeypatch.setattr(qr, "_get_chain", lambda: FakeChain())

    history = [
        {"role": "user", "content": "How fast is the RAG pipeline?"},
        {"role": "assistant", "content": "About 15-25 seconds per response."},
    ]
    result = qr.rewrite_question("What about its latency?", history)
    assert result == "What is the retrieval latency of the RAG pipeline?"


def test_rewrite_question_falls_back_on_llm_error(monkeypatch):
    class FailingChain:
        def invoke(self, inputs):
            raise RuntimeError("model unavailable")

    monkeypatch.setattr(qr, "_get_chain", lambda: FailingChain())

    history = [{"role": "user", "content": "hi"}]
    original = "What about it?"
    result = qr.rewrite_question(original, history)
    assert result == original
