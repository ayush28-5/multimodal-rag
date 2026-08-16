import gc
import ollama
import chromadb
from src.config import (
    CHROMA_DB_DIR, COLLECTION_NAME,
    EMBEDDING_MODEL, TOP_K_RESULTS
)


def get_embedding(text):
    """Get embedding for a single text using Ollama."""
    response = ollama.embed(model=EMBEDDING_MODEL, input=text)
    return response["embeddings"][0]


class VectorStoreRepository:
    """Repository pattern for ChromaDB access."""

    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def add_items(self, ids, documents, embeddings, metadatas):
        """Add or update items in the collection (upsert prevents duplicates)."""
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def query(self, query_embedding, n_results=TOP_K_RESULTS):
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

    def count(self):
        return self.collection.count()

    def clear(self):
        """Delete and recreate the collection."""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )


# Initialize global repository instance
repo = VectorStoreRepository()


def store_chunks(text_chunks, captioned_images):
    """Embed and store all text chunks and captioned images with deduplication."""
    all_items = []
    counter = 0
    seen_texts = set()
    duplicates_skipped = 0

    # Process text chunks
    for chunk in text_chunks:
        text_hash = hash(chunk["text"].strip().lower())
        if text_hash in seen_texts:
            duplicates_skipped += 1
            continue
        seen_texts.add(text_hash)

        all_items.append({
            "id": f"chunk_{counter}",
            "text": chunk["text"],
            "metadata": {
                "content_type": chunk["content_type"],
                "chunk_type": chunk["content_type"],
                "page": chunk["page"],
                "source": chunk["source_file"],
                "visual_content_path": ""
            }
        })
        counter += 1

    # Process captioned images
    for i, img in enumerate(captioned_images):
        text_hash = hash(img["text"].strip().lower())
        if text_hash in seen_texts:
            duplicates_skipped += 1
            continue
        seen_texts.add(text_hash)

        all_items.append({
            "id": f"chunk_{counter}",
            "text": img["text"],
            "metadata": {
                "content_type": img["content_type"],
                "chunk_type": img["content_type"],
                "page": img["page"],
                "source": img["source_file"],
                "visual_content_path": img["path"]
            }
        })
        counter += 1

    if duplicates_skipped > 0:
        print(f"   Deduplication: skipped {duplicates_skipped} duplicate text chunk(s)")

    # Embed and store in batches
    batch_size = 10
    total = len(all_items)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = all_items[start:end]

        ids = [item["id"] for item in batch]
        texts = [item["text"] for item in batch]
        metadatas = [item["metadata"] for item in batch]

        print(f"   Embedding batch {start//batch_size + 1}/{(total + batch_size - 1)//batch_size}")

        embeddings = [get_embedding(text) for text in texts]

        repo.add_items(ids, texts, embeddings, metadatas)

        # Free batch memory
        del embeddings, batch

    # Final memory cleanup
    gc.collect()

    print(f"\n   Stored {total} unique items in ChromaDB (total: {repo.count()})")


def query_collection(query_text, n_results=TOP_K_RESULTS):
    """Query ChromaDB and return top matching results."""
    query_embedding = get_embedding(query_text)
    return repo.query(query_embedding, n_results)