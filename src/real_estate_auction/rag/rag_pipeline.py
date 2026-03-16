import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

PROPERTIES_DIR = "data/raw_properties"
CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "properties"


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)


def chunk_markdown(text: str, property_id: str) -> list[dict]:
    """
    Split a property markdown file into chunks by section (## headings).
    Each section becomes one chunk so retrieval returns focused content
    rather than an entire document. The property header is prepended to
    every chunk so each one is self-contained.
    """
    lines = text.strip().split("\n")

    # Grab the header block (everything before the first ## section)
    header_lines = []
    for line in lines:
        if line.startswith("## "):
            break
        header_lines.append(line)
    header = "\n".join(header_lines).strip()

    # Split into sections at each ## heading
    sections = []
    current_section = []
    for line in lines:
        if line.startswith("## ") and current_section:
            sections.append("\n".join(current_section).strip())
            current_section = [line]
        else:
            current_section.append(line)
    if current_section:
        sections.append("\n".join(current_section).strip())

    chunks = []
    for i, section in enumerate(sections):
        # Skip the header block itself — it's repeated as context in each chunk
        if section == header:
            continue
        chunks.append({
            "id": f"{property_id}_chunk_{i}",
            "text": f"{header}\n\n{section}",
            "property_id": property_id,
        })

    return chunks


def index_properties():
    collection = get_collection()

    already_indexed = set()
    if collection.count() > 0:
        existing = collection.get()
        for meta in existing["metadatas"]:
            already_indexed.add(meta["property_id"])

    files = list(Path(PROPERTIES_DIR).glob("*.md"))
    new_files = [f for f in files if f.stem not in already_indexed]

    if not new_files:
        print("All properties already indexed.")
        return

    for filepath in new_files:
        text = filepath.read_text(encoding="utf-8")
        property_id = filepath.stem
        chunks = chunk_markdown(text, property_id)

        collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[{"property_id": c["property_id"]} for c in chunks],
        )
        print(f"Indexed {property_id} ({len(chunks)} chunks)")

    print(f"Done. Total documents in index: {collection.count()}")


def search(query: str, n_results: int = 5) -> list[dict]:
    """
    Return up to n_results *distinct* properties ranked by relevance.
    """
    collection = get_collection()
    
    fetch_count = min(n_results * 5, collection.count() or 1)
    results = collection.query(query_texts=[query], n_results=fetch_count)

    matches = []
    seen: set[str] = set()
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        pid = meta["property_id"]
        if pid not in seen:
            seen.add(pid)
            matches.append({"property_id": pid, "content": doc})
        if len(matches) >= n_results:
            break

    return matches


if __name__ == "__main__":
    index_properties()