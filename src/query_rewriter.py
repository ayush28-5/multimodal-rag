"""
Query Rewriter — converts ambiguous follow-up questions into standalone questions.
Uses a cached LLM instance for speed.
"""
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import ENABLE_PROFILING
from src.llm_provider import get_rewriter_llm


ENABLE_QUERY_REWRITING = True

REWRITER_WINDOW_MESSAGES = 6

# Pronouns / references that signal a follow-up needing rewrite
FOLLOWUP_TOKENS = {
    "it", "its", "they", "them", "their", "this", "that", "these", "those",
    "previous", "above", "earlier", "first", "second", "last", "next",
    "former", "latter", "such",
}


REWRITER_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """Rewrite the user's question into a standalone version using the conversation.
Rules:
- Resolve pronouns (it, they, this, that) using earlier turns.
- Keep intent unchanged.
- If already complete, return it as-is.
- Output ONLY the rewritten question. No quotes, no preamble."""),
    ("human", """Conversation:
{history}

Latest question: {question}

Standalone question:""")
])


_chain = None


def _get_chain():
    global _chain
    if _chain is None:
        _chain = REWRITER_TEMPLATE | get_rewriter_llm() | StrOutputParser()
    return _chain


def _format_history(messages, max_messages=REWRITER_WINDOW_MESSAGES):
    if not messages:
        return "(none)"
    recent = messages[-max_messages:] if len(messages) > max_messages else messages
    lines = []
    for m in recent:
        role = "User" if m.get("role") == "user" else "Assistant"
        c = (m.get("content") or "").strip()
        if c:
            # Truncate very long assistant messages for the rewriter
            if len(c) > 300:
                c = c[:300] + "..."
            lines.append(f"{role}: {c}")
    return "\n".join(lines) if lines else "(none)"


def _needs_rewrite(question):
    """Heuristic: only call the LLM if the question likely contains references."""
    q = question.lower().strip()
    # Very short questions almost always need context
    if len(q.split()) <= 4:
        return True
    # Check for pronouns / references
    tokens = set(q.replace("?", "").replace(",", "").split())
    return bool(tokens & FOLLOWUP_TOKENS)


def _clean_output(raw, original):
    if not raw:
        return original
    text = raw.strip()
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    for prefix in ["rewritten:", "standalone question:", "answer:", "result:"]:
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()
            break
    if "\n" in text:
        text = text.split("\n")[0].strip()
    if not text or len(text) > len(original) * 6:
        return original
    return text


def rewrite_question(question, conversation_history):
    """Rewrite a follow-up question into a standalone question."""
    if not ENABLE_QUERY_REWRITING:
        return question

    question = (question or "").strip()
    if not question or not conversation_history:
        return question

    # Skip rewriter entirely if the question doesn't look like a follow-up
    if not _needs_rewrite(question):
        if ENABLE_PROFILING:
            print(f"   [Rewriter] Skipped (no references): {question}")
        return question

    t0 = time.time()
    try:
        history_text = _format_history(conversation_history)
        chain = _get_chain()
        raw = chain.invoke({"history": history_text, "question": question})
        rewritten = _clean_output(raw, question)

        if ENABLE_PROFILING:
            dt = time.time() - t0
            if rewritten != question:
                print(f"   [Rewriter] {dt:.2f}s | {question} → {rewritten}")
            else:
                print(f"   [Rewriter] {dt:.2f}s | unchanged")
        return rewritten
    except Exception as e:
        print(f"   [Rewriter] Error: {e}")
        return question