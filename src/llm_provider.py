"""
LLM provider — caches LLM instances so we don't recreate them on every query.
"""
from langchain_ollama import OllamaLLM
from src.config import (
    LLM_MODEL, REWRITER_MODEL,
    LLM_NUM_CTX, LLM_NUM_PREDICT, LLM_TEMPERATURE,
    LLM_TOP_P, LLM_TOP_K, LLM_REPEAT_PENALTY,
)


_llm_instance = None
_rewriter_instance = None


def get_llm():
    """Get the cached main LLM instance (for answering questions)."""
    global _llm_instance
    if _llm_instance is None:
        print(f"   [LLM] Initializing {LLM_MODEL} (cached)...")
        _llm_instance = OllamaLLM(
            model=LLM_MODEL,
            num_ctx=LLM_NUM_CTX,
            num_predict=LLM_NUM_PREDICT,
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            top_k=LLM_TOP_K,
            repeat_penalty=LLM_REPEAT_PENALTY,
        )
    return _llm_instance


def get_rewriter_llm():
    """Get the cached rewriter LLM (smaller, deterministic)."""
    global _rewriter_instance
    if _rewriter_instance is None:
        print(f"   [LLM] Initializing rewriter {REWRITER_MODEL} (cached)...")
        _rewriter_instance = OllamaLLM(
            model=REWRITER_MODEL,
            num_ctx=1024,
            num_predict=80,
            temperature=0.0,
        )
    return _rewriter_instance


def warmup_models():
    """Pre-load models at app startup to avoid first-query latency."""
    print("   [LLM] Warming up models...")
    get_llm()
    get_rewriter_llm()
    print("   [LLM] Models ready.")