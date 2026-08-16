"""
Integration test — picks up any file in the data folder automatically.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser import parse_all_documents
from src.captioner import caption_all_images
from src.embedder import store_chunks, repo
from src.retriever import retrieve_and_answer


def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    if not os.path.exists(data_dir):
        print(f"ERROR: 'data' folder not found")
        sys.exit(1)

    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

    if not files:
        print("ERROR: No files found in data folder. Drop a file first.")
        sys.exit(1)

    print(f"\nFound {len(files)} file(s) in data folder:")
    for f in files:
        size_kb = os.path.getsize(os.path.join(data_dir, f)) / 1024
        print(f"  - {f} ({size_kb:.1f} KB)")

    # ===== STAGE 1: PARSING =====
    print("\n" + "=" * 60)
    print("STAGE 1: PARSING")
    print("=" * 60)

    text_chunks, images = parse_all_documents()

    text_count = sum(1 for c in text_chunks if c["content_type"] == "text")
    table_count = sum(1 for c in text_chunks if c["content_type"] == "table")
    image_count = sum(1 for img in images if img["content_type"] == "image")
    flowchart_count = sum(1 for img in images if img["content_type"] == "flowchart")

    print(f"\nParsing summary:")
    print(f"  Text chunks: {text_count}")
    print(f"  Tables: {table_count}")
    print(f"  Images: {image_count}")
    print(f"  Flowcharts: {flowchart_count}")

    # ===== STAGE 2: CAPTIONING =====
    captioned_images = []
    if images:
        print("\n" + "=" * 60)
        print("STAGE 2: CAPTIONING")
        print("=" * 60)
        captioned_images = caption_all_images(images)

    # ===== STAGE 3: STORAGE =====
    print("\n" + "=" * 60)
    print("STAGE 3: EMBEDDING & STORAGE")
    print("=" * 60)

    try:
        repo.clear()
        print("  Cleared existing collection")
    except Exception:
        pass

    store_chunks(text_chunks, captioned_images)

    # ===== STAGE 4: QUERY LOOP =====
    print("\n" + "=" * 60)
    print("STAGE 4: QUERY INTERFACE")
    print("=" * 60)
    print("\nType your question and press Enter. Type 'exit' to stop.\n")

    while True:
        try:
            query = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("Exiting.")
            break

        print("\nGenerating response...\n")
        try:
            answer, sources = retrieve_and_answer(query)

            print("-" * 60)
            print("ANSWER:")
            print("-" * 60)
            print(answer)
            print()

            if sources:
                print("-" * 60)
                print(f"SOURCES ({len(sources)}):")
                print("-" * 60)
                for i, src in enumerate(sources):
                    relevance = (1 - src["distance"]) * 100
                    print(f"  [{i+1}] {src['content_type']} on page {src['page']} from {src['source']} (relevance: {relevance:.0f}%)")
                print()
        except Exception as e:
            print(f"ERROR: {e}\n")


if __name__ == "__main__":
    main()