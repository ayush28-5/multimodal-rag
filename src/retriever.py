"""
Retriever — pulls relevant chunks from ChromaDB, builds a compact prompt,
and generates a grounded answer using the cached Phi LLM.

Supports streaming via stream_answer() generator.
"""
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import (
    TOP_K_CANDIDATES, TOP_K_RESULTS, MAX_DISTANCE,
    ENABLE_PROFILING, ENABLE_STREAMING,
)
from src.embedder import query_collection
from src.llm_provider import get_llm


# ============================================
# COMPACT PROMPT — tuned for Phi
# ============================================
SYSTEM_PROMPT = """You are a document assistant. Answer using ONLY the context below.

Rules:
- Be concise. Use 2-5 sentences unless a list is needed.
- Quote facts directly from the context.
- If the context doesn't contain the answer, say: "I don't have enough information in the provided documents to answer this question."
- Don't apologize or hedge. Don't invent information."""


ANSWER_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """{history_block}Context:
{context}

Question: {question}

Answer:""")
])


_chain = None


def _get_chain():
    global _chain
    if _chain is None:
        _chain = ANSWER_TEMPLATE | get_llm() | StrOutputParser()
    return _chain


# ============================================
# CONTEXT BUILDING — dedup + trim
# ============================================
def _dedup_chunks(docs, metas, dists):
    """Remove near-duplicate chunks (same source+page or identical text prefixes)."""
    seen_keys = set()
    seen_prefixes = set()
    kept = []

    for i in range(len(docs)):
        doc = docs[i]
        meta = metas[i]
        dist = dists[i]

        # Drop chunks above distance threshold
        if dist > MAX_DISTANCE:
            continue

        # Dedup by source+page
        key = (meta.get("source", ""), meta.get("page", ""), meta.get("content_type", ""))
        if key in seen_keys:
            continue
        seen_keys.add(key)

        # Dedup by content prefix (catches overlapping chunks)
        prefix = doc[:100].strip().lower()
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)

        kept.append((doc, meta, dist))

    return kept


def _build_context(results):
    """Build a compact context block + structured sources list."""
    docs_list = results["documents"][0]
    meta_list = results["metadatas"][0]
    dist_list = results["distances"][0]

    deduped = _dedup_chunks(docs_list, meta_list, dist_list)
    deduped = deduped[:TOP_K_RESULTS]

    context_parts = []
    sources = []

    for i, (doc, metadata, distance) in enumerate(deduped):
        content_type = metadata.get("content_type", "text")
        page = metadata.get("page", "?")
        source = metadata.get("source", "")
        # Compact source label
        label = f"[{i+1}] {content_type} p.{page}"
        context_parts.append(f"{label}\n{doc}")

        sources.append({
            "content_type": content_type,
            "page": page,
            "source": source,
            "distance": distance,
            "visual_content_path": metadata.get("visual_content_path", ""),
            "text": doc,
        })

    context = "\n\n".join(context_parts)
    return context, sources


def _format_history_block(history_text):
    """Format conversation history compactly, or empty string if none."""
    if not history_text or history_text in ("(none)", "(no previous conversation)"):
        return ""
    return f"Recent conversation:\n{history_text}\n\n"


# ============================================
# NON-STREAMING ANSWER (compatible with existing app)
# ============================================
def retrieve_and_answer(query, conversation_history_text="(none)"):
    """
    Retrieve + generate a complete answer (non-streaming).
    Returns (answer, sources).
    """
    timings = {}

    # === Retrieval ===
    t0 = time.time()
    results = query_collection(query, n_results=TOP_K_CANDIDATES)
    timings["retrieval"] = time.time() - t0

    docs = results.get("documents", [])
    if not docs or not docs[0]:
        return "No documents have been ingested yet. Please upload and process documents first.", []

    # === Context ===
    t0 = time.time()
    context, sources = _build_context(results)
    history_block = _format_history_block(conversation_history_text)
    timings["context"] = time.time() - t0

    if not sources:
        # All retrieved chunks were filtered out as irrelevant
        return "I don't have enough information in the provided documents to answer this question.", []

    # === LLM Inference ===
    t0 = time.time()
    chain = _get_chain()
    answer = chain.invoke({
        "history_block": history_block,
        "context": context,
        "question": query,
    }).strip()
    timings["llm"] = time.time() - t0

    if ENABLE_PROFILING:
        total = sum(timings.values())
        print(
            f"   [Retriever] retrieval={timings['retrieval']:.2f}s "
            f"context={timings['context']:.2f}s "
            f"llm={timings['llm']:.2f}s "
            f"total={total:.2f}s "
            f"chunks={len(sources)}"
        )

    return answer, sources


# ============================================
# STREAMING ANSWER (generator yielding partial tokens)
# ============================================
def stream_answer(query, conversation_history_text="(none)"):
    """
    Stream tokens as they're generated.
    Yields:
        ("token", str)        — partial text chunk
        ("sources", list)     — final sources list when done
        ("error", str)        — if anything goes wrong
    """
    timings = {}

    t0 = time.time()
    results = query_collection(query, n_results=TOP_K_CANDIDATES)
    timings["retrieval"] = time.time() - t0

    docs = results.get("documents", [])
    if not docs or not docs[0]:
        yield ("token", "No documents have been ingested yet. Please upload and process documents first.")
        yield ("sources", [])
        return

    t0 = time.time()
    context, sources = _build_context(results)
    history_block = _format_history_block(conversation_history_text)
    timings["context"] = time.time() - t0

    if not sources:
        yield ("token", "I don't have enough information in the provided documents to answer this question.")
        yield ("sources", [])
        return

    t0 = time.time()
    try:
        llm = get_llm()
        prompt_value = ANSWER_TEMPLATE.format_prompt(
            history_block=history_block,
            context=context,
            question=query,
        )
        prompt_str = prompt_value.to_string()

        # Use Ollama's native streaming
        for chunk in llm.stream(prompt_str):
            if chunk:
                yield ("token", chunk)
    except Exception as e:
        yield ("error", str(e))
        return
    timings["llm"] = time.time() - t0

    yield ("sources", sources)

    if ENABLE_PROFILING:
        total = sum(timings.values())
        print(
            f"   [Retriever-Stream] retrieval={timings['retrieval']:.2f}s "
            f"context={timings['context']:.2f}s "
            f"llm={timings['llm']:.2f}s "
            f"total={total:.2f}s "
            f"chunks={len(sources)}"
        )