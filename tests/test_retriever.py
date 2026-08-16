"""
Unit tests for src/retriever.py

Tests the pure dedup/context-building logic. No ChromaDB or LLM calls
are made — query_collection and get_llm are never touched by these tests.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retriever import _dedup_chunks, _build_context, _format_history_block
from src.config import MAX_DISTANCE, TOP_K_RESULTS


# ============================================
# _dedup_chunks
# ============================================

def test_dedup_chunks_drops_above_distance_threshold():
    docs = ["relevant chunk text here"]
    metas = [{"source": "a.pdf", "page": 1, "content_type": "text"}]
    dists = [MAX_DISTANCE + 0.1]
    assert _dedup_chunks(docs, metas, dists) == []


def test_dedup_chunks_keeps_below_threshold():
    docs = ["relevant chunk text here"]
    metas = [{"source": "a.pdf", "page": 1, "content_type": "text"}]
    dists = [MAX_DISTANCE - 0.1]
    kept = _dedup_chunks(docs, metas, dists)
    assert len(kept) == 1


def test_dedup_chunks_removes_same_source_page_type_duplicates():
    docs = ["first version of the text", "second version of the text"]
    metas = [
        {"source": "a.pdf", "page": 1, "content_type": "text"},
        {"source": "a.pdf", "page": 1, "content_type": "text"},
    ]
    dists = [0.1, 0.2]
    kept = _dedup_chunks(docs, metas, dists)
    assert len(kept) == 1
    assert kept[0][0] == "first version of the text"


def test_dedup_chunks_removes_overlapping_prefix_duplicates():
    shared_prefix = "a" * 100
    docs = [shared_prefix + " tail one", shared_prefix + " tail two"]
    metas = [
        {"source": "a.pdf", "page": 1, "content_type": "text"},
        {"source": "b.pdf", "page": 2, "content_type": "text"},  # different key
    ]
    dists = [0.1, 0.1]
    kept = _dedup_chunks(docs, metas, dists)
    # Even though source/page differ, identical first-100-char prefix should dedup
    assert len(kept) == 1


def test_dedup_chunks_keeps_distinct_chunks():
    docs = ["alpha chunk text", "beta chunk text completely different"]
    metas = [
        {"source": "a.pdf", "page": 1, "content_type": "text"},
        {"source": "a.pdf", "page": 2, "content_type": "text"},
    ]
    dists = [0.1, 0.2]
    kept = _dedup_chunks(docs, metas, dists)
    assert len(kept) == 2


# ============================================
# _build_context
# ============================================

def _make_results(docs, metas, dists):
    return {
        "documents": [docs],
        "metadatas": [metas],
        "distances": [dists],
    }


def test_build_context_returns_labeled_context_and_sources():
    results = _make_results(
        docs=["Some retrieved text about RAG."],
        metas=[{"source": "doc.pdf", "page": 2, "content_type": "text"}],
        dists=[0.1],
    )
    context, sources = _build_context(results)

    assert "[1] text p.2" in context
    assert "Some retrieved text about RAG." in context
    assert len(sources) == 1
    assert sources[0]["source"] == "doc.pdf"
    assert sources[0]["page"] == 2
    assert sources[0]["distance"] == 0.1


def test_build_context_respects_top_k_results():
    n = TOP_K_RESULTS + 5
    docs = [f"chunk {i} with unique content here" for i in range(n)]
    metas = [{"source": f"doc{i}.pdf", "page": i, "content_type": "text"} for i in range(n)]
    dists = [0.1] * n

    results = _make_results(docs, metas, dists)
    context, sources = _build_context(results)

    assert len(sources) == TOP_K_RESULTS


def test_build_context_filters_out_of_range_distance_chunks():
    results = _make_results(
        docs=["irrelevant far chunk"],
        metas=[{"source": "doc.pdf", "page": 1, "content_type": "text"}],
        dists=[MAX_DISTANCE + 1.0],
    )
    context, sources = _build_context(results)
    assert context == ""
    assert sources == []


# ============================================
# _format_history_block
# ============================================

def test_format_history_block_empty_for_none_marker():
    assert _format_history_block("(none)") == ""
    assert _format_history_block("") == ""
    assert _format_history_block(None) == ""


def test_format_history_block_wraps_real_history():
    result = _format_history_block("User: hi\nAssistant: hello")
    assert result.startswith("Recent conversation:\n")
    assert "User: hi" in result
    assert result.endswith("\n\n")
