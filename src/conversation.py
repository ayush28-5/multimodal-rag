"""
Conversation memory module.
Handles in-session chat history and prompt formatting.

NO persistence — memory exists only in Streamlit's session state.
NO rewriting logic (that lives in src/query_rewriter.py).
"""
from datetime import datetime


# Sliding window: max number of EXCHANGES (user + assistant pairs) sent to LLM
MAX_WINDOW_EXCHANGES = 10


# ============================================
# MESSAGE BUILDERS
# ============================================

def make_user_message(content):
    """Build a user message dict."""
    return {
        "role": "user",
        "content": content,
        "time": datetime.now().strftime("%H:%M")
    }


def make_assistant_message(content, sources=None, confidence="high", generation_time=0.0):
    """Build an assistant message dict."""
    return {
        "role": "assistant",
        "content": content,
        "sources": sources or [],
        "confidence": confidence,
        "generation_time": generation_time,
        "time": datetime.now().strftime("%H:%M")
    }


# ============================================
# CONVERSATION WINDOW
# ============================================

def get_recent_history(messages, max_exchanges=MAX_WINDOW_EXCHANGES):
    """
    Return the most recent N exchanges from the conversation.
    One exchange = one user + one assistant message.
    """
    if not messages:
        return []

    max_messages = max_exchanges * 2

    if len(messages) <= max_messages:
        return list(messages)

    return list(messages[-max_messages:])


def format_history_for_prompt(messages, max_exchanges=MAX_WINDOW_EXCHANGES):
    """
    Format the sliding-window history as a transcript for the LLM prompt.
    """
    recent = get_recent_history(messages, max_exchanges)

    if not recent:
        return "(no previous conversation)"

    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = (msg.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")

    return "\n\n".join(lines) if lines else "(no previous conversation)"


# ============================================
# SESSION MANAGEMENT
# ============================================

def clear_conversation(session_state):
    """
    Reset all conversation-related session state.
    Documents and ChromaDB are NEVER touched here.
    """
    session_state.messages = []
    session_state.rewrite_cache = {}
    session_state.last_query = None