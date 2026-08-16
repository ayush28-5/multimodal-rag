import sys
from src.parser import parse_all_documents
from src.captioner import caption_all_images
from src.embedder import store_chunks, repo
from src.retriever import retrieve_and_answer, stream_answer


def ingest_documents():
    """Full pipeline: parse → caption → embed → store."""
    print("\n" + "=" * 50)
    print("STEP 1: Parsing documents")
    print("=" * 50)
    text_chunks, images = parse_all_documents()

    if not text_chunks and not images:
        print("No content found to ingest.")
        return

    flowcharts = sum(1 for img in images if img["content_type"] == "flowchart")
    regular = len(images) - flowcharts
    tables = sum(1 for tc in text_chunks if tc["content_type"] == "table")
    texts = len(text_chunks) - tables

    print(f"\n   Found: {texts} text chunks, {regular} images, {flowcharts} flowcharts, {tables} tables")

    print("\n" + "=" * 50)
    print("STEP 2: Captioning visuals with VLM")
    print("=" * 50)
    captioned_images = caption_all_images(images) if images else []

    print("\n" + "=" * 50)
    print("STEP 3: Embedding and storing in ChromaDB")
    print("=" * 50)
    store_chunks(text_chunks, captioned_images)

    print("\n" + "=" * 50)
    print(f"INGESTION COMPLETE — {repo.count()} items in DB")
    print("=" * 50)


def ask(query, conversation_history_text="(none)"):
    """Non-streaming answer. Returns (answer, sources)."""
    return retrieve_and_answer(query, conversation_history_text)


def ask_stream(query, conversation_history_text="(none)"):
    """Streaming answer generator. Yields ('token', str) | ('sources', list) | ('error', str)."""
    return stream_answer(query, conversation_history_text)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        ingest_documents()
    else:
        print("Usage: python -m src.rag ingest")